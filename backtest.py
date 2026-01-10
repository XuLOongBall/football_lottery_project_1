# -*- coding: utf-8 -*-
"""
backtest.py

按月回测（不泄露未来） + 联赛历史样本门槛筛选（不足回填补齐）。

核心约束：
- 对于某个月 m，只能使用该月第一场比赛时间之前的数据估计参数/统计联赛样本量；
- 取消/无效场次（is_void=1）按退票处理：profit=0、hit=NaN、倍投层级不变；
- 每月下注场次数由 config.MONTHLY_MIN_BETS（若无则默认16）控制；
- 联赛历史样本门槛由 config.MIN_LEAGUE_HIST（若无则默认200）控制。
"""
from __future__ import annotations

from typing import Any, Dict, List
import ast

import pandas as pd
import numpy as np

import config
from selection import select_matches_for_month
from betting import settle_bet


def _import_predict_goal_distribution():
    # 兼容不同项目结构：model.poisson_model 或 poisson_model
    try:
        from model.poisson_model import predict_goal_distribution  # type: ignore
        return predict_goal_distribution
    except Exception:
        try:
            from poisson_model import predict_goal_distribution  # type: ignore
            return predict_goal_distribution
        except Exception:
            return None


_PREDICT = _import_predict_goal_distribution()


def _predict_poisson_fallback(history_goals: np.ndarray, max_goal: int = 7) -> Dict[int, float]:
    """
    不依赖 scipy 的泊松分布预测（fallback）。
    """
    history_goals = np.asarray(history_goals, dtype=float)
    history_goals = history_goals[~np.isnan(history_goals)]
    if len(history_goals) == 0:
        raise ValueError("history_goals is empty")
    lam = float(history_goals.mean())

    probs: Dict[int, float] = {}
    # 0..6
    for k in range(max_goal):
        # poisson pmf: e^{-lam} lam^k / k!
        probs[k] = float(np.exp(-lam) * (lam ** k) / float(np.math.factorial(k)))
    probs[max_goal] = float(max(0.0, 1.0 - sum(probs.values())))
    return probs


def _ensure_odds_dict(x: Any) -> Dict[int, float]:
    if isinstance(x, dict):
        d = x
    elif isinstance(x, str):
        try:
            d = ast.literal_eval(x)
        except Exception:
            return {}
    else:
        return {}
    out: Dict[int, float] = {}
    for k, v in d.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def run_backtest(df: pd.DataFrame, scheme: str = "P") -> List[Dict[str, Any]]:
    """
    主回测流程：按月选场 + 按时间结算
    scheme="EV" -> 方案A（按EV排序选场）
    scheme="P"  -> 方案B（按p_S排序选场）
    """
    if df is None or len(df) == 0:
        return []

    df = df.copy()

    # 基础字段统一
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    if "month_id" not in df.columns:
        df["month_id"] = df["date"].dt.strftime("%Y-%m")

    if "is_void" not in df.columns:
        df["is_void"] = 0
    df["is_void"] = pd.to_numeric(df["is_void"], errors="coerce").fillna(0).astype(int)

    # league 字段兼容
    if "league" not in df.columns:
        if "联赛" in df.columns:
            df["league"] = df["联赛"].astype(str)
        else:
            df["league"] = "UNKNOWN"

    # teams 字段兼容
    if "home_team" not in df.columns:
        if "主队" in df.columns:
            df["home_team"] = df["主队"].astype(str)
        else:
            df["home_team"] = "HOME"
    if "away_team" not in df.columns:
        if "客队" in df.columns:
            df["away_team"] = df["客队"].astype(str)
        else:
            df["away_team"] = "AWAY"

    # total_goals 字段兼容
    if "total_goals" not in df.columns:
        if "总进球" in df.columns:
            df["total_goals"] = pd.to_numeric(df["总进球"], errors="coerce")
        else:
            df["total_goals"] = np.nan

    # odds
    if "odds" not in df.columns:
        raise KeyError("cleaned df must contain 'odds' column (dict of odds per goal count).")

    df = df.sort_values("date").reset_index(drop=True)

    monthly_min_bets = int(getattr(config, "MONTHLY_MIN_BETS", 16))
    rolling_window = int(getattr(config, "ROLLING_WINDOW", 200))
    min_league_hist = int(getattr(config, "MIN_LEAGUE_HIST", 200))

    lookback_days = getattr(config, "LOOKBACK_DAYS", None)
    if isinstance(lookback_days, str) and lookback_days.upper() == "ALL":
        lookback_days = None
    if lookback_days is not None:
        try:
            lookback_days = int(lookback_days)
        except Exception:
            lookback_days = None

    results: List[Dict[str, Any]] = []
    bankroll = 0.0
    martingale_level = 0

    for month_id in sorted(df["month_id"].dropna().unique()):
        df_month = df[df["month_id"] == month_id].copy()
        if len(df_month) == 0:
            continue
        df_month = df_month.sort_values("date")
        month_start = df_month["date"].min()

        # 历史：严格用月初之前，且只用已结算（非 void）
        df_history = df[(df["date"] < month_start) & (df["is_void"] == 0)].copy()

        # 可选：只用最近 lookback_days
        if lookback_days is not None:
            cutoff = month_start - pd.Timedelta(days=int(lookback_days))
            df_history = df_history[df_history["date"] >= cutoff].copy()

        history_goals = df_history["total_goals"].dropna().astype(int).values
        if len(history_goals) == 0:
            # 没历史就跳过（最早期可能发生）
            continue

        # 只取最近 rolling_window 场
        history_goals = history_goals[-rolling_window:]

        # 预测总进球分布：
        # - 默认使用“全局（跨联赛）”历史窗口作为 fallback
        # - 若某个联赛在月初之前历史样本数 >= min_league_hist，则为该联赛单独估计分布（仍只用历史，不泄露未来）
        def _predict(goals_arr):
            if _PREDICT is not None:
                try:
                    return _PREDICT(goals_arr, max_goal=7)
                except TypeError:
                    return _PREDICT(goals_arr)
                except Exception:
                    return _predict_poisson_fallback(goals_arr, max_goal=7)
            return _predict_poisson_fallback(goals_arr, max_goal=7)

        global_prob_dist = _predict(history_goals)

        # 联赛历史样本量（同样只用 df_history）
        league_counts = df_history["league"].astype(str).value_counts().to_dict()

        # 为“历史样本足够”的联赛预先缓存各自的分布
        league_prob_cache: Dict[str, Dict[int, float]] = {}
        if min_league_hist > 0:
            for lg, cnt in league_counts.items():
                if int(cnt) < min_league_hist:
                    continue
                lg_goals = df_history[df_history["league"].astype(str) == str(lg)]["total_goals"].dropna().astype(int).values
                if len(lg_goals) == 0:
                    continue
                lg_goals = lg_goals[-rolling_window:]
                try:
                    league_prob_cache[str(lg)] = _predict(lg_goals)
                except Exception:
                    # 单个联赛估计失败就跳过，仍用全局 fallback
                    continue

        # 构造本月比赛列表（供 selection 评分排序）
        matches: List[Dict[str, Any]] = []
        for _, r in df_month.iterrows():
            league = str(r.get("league", "UNKNOWN"))
            odds = _ensure_odds_dict(r.get("odds"))
            match = {
                "date": r["date"],
                "month_id": month_id,
                "league": league,
                "home_team": str(r.get("home_team", "")),
                "away_team": str(r.get("away_team", "")),
                "odds": odds,
                "prob_dist": league_prob_cache.get(league, global_prob_dist),
                "prob_scope": ("league" if league in league_prob_cache else "global"),
                "is_void": int(r.get("is_void", 0)),
                "y_true": r.get("total_goals", np.nan),
                "league_hist_count": int(league_counts.get(league, 0)),
            }
            matches.append(match)

        selected = select_matches_for_month(
            matches=matches,
            min_bets=monthly_min_bets,
            scheme=scheme,
            min_league_hist=min_league_hist
        )

        # 结算顺序按时间（避免“同月内按score顺序”带来的顺序偏差）
        selected = sorted(selected, key=lambda x: x.get("date"))

        for match in selected:
            is_void = int(match.get("is_void", 0)) == 1
            y_true = match.get("y_true", np.nan)

            if is_void or pd.isna(y_true):
                settle = {
                    "stake_total": 0.0,
                    "stake_split": {},
                    "hit": pd.NA,
                    "profit": 0.0,
                    "next_level": martingale_level,
                    "stoploss": 0,
                }
            else:
                settle = settle_bet(
                    selected_set=match["selected_set"],
                    odds=match["odds"],
                    y_true=int(y_true),
                    level=martingale_level
                )

            martingale_level = int(settle["next_level"])
            bankroll += float(settle["profit"])

            results.append({
                "date": match["date"],
                "month_id": match["month_id"],
                "scheme": match.get("scheme", scheme),
                "ev_fallback": match.get("ev_fallback", 0),

                "league": match.get("league"),
                "league_hist_count": match.get("league_hist_count", 0),
                "prob_scope": match.get("prob_scope", "global"),
                "eligible_league": match.get("eligible_league", 0),
                "filled_from_low_hist": match.get("filled_from_low_hist", 0),

                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "is_void": int(match.get("is_void", 0)),
                "y_true": (np.nan if pd.isna(y_true) else int(y_true)),

                "selected_set": match.get("selected_set"),
                "p_selected": match.get("p_selected"),
                "ev_selected": match.get("ev_selected"),
                "score": match.get("score"),

                "stake_total": settle["stake_total"],
                "stake_split": settle["stake_split"],
                "hit": settle["hit"],
                "profit": settle["profit"],
                "stoploss": settle["stoploss"],
                "bankroll": bankroll,
                "martingale_level": martingale_level,
            })

    return results
