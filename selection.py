# -*- coding: utf-8 -*-
"""
selection.py

月度选场 + 单场选集合逻辑（支持联赛历史样本门槛筛选，并保证不足时回填补满）。

关键点（不泄露未来）：
- 联赛历史样本数 league_hist_count 必须由“当月第一场比赛之前”的历史数据统计得到；
- 当月选场时：优先选 league_hist_count >= min_league_hist 的比赛；
  若不足 min_bets，则从低样本联赛中按 score 继续补齐到 min_bets。

该文件不依赖贝叶斯/MCMC。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Optional
import math

import config


def _to_int_key_odds(odds: Any) -> Dict[int, float]:
    """
    odds 可能是 {0:1.9, ...} / {"0":"1.9", ...} / 或者含不可用项。
    统一转换为 Dict[int, float]，丢弃无法转换的键值。
    """
    if not isinstance(odds, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in odds.items():
        try:
            ki = int(k)
            vf = float(v)
            out[ki] = vf
        except Exception:
            continue
    return out


def _p(prob_dist: Any, k: int) -> float:
    """
    prob_dist 既可能是 dict，也可能是 list/np.array。
    """
    try:
        if isinstance(prob_dist, dict):
            return float(prob_dist.get(k, 0.0))
        return float(prob_dist[k])
    except Exception:
        return 0.0


def compute_p_selected(prob_dist: Any, selected_set: Iterable[int]) -> float:
    return float(sum(_p(prob_dist, int(k)) for k in selected_set))


def compute_ev(prob_dist: Any, odds: Dict[int, float], selected_set: Iterable[int]) -> float:
    """
    简化 EV（与资金分配方式无关的“相对收益”指标）：
    - 这里按“等额分配”口径：平均命中收益 - 1
    """
    selected = list(selected_set)
    if len(selected) == 0:
        return -math.inf

    # 若缺赔率，则该集合不可下注
    for k in selected:
        if int(k) not in odds:
            return -math.inf

    m = len(selected)
    return float(sum(_p(prob_dist, int(k)) * float(odds[int(k)]) for k in selected) / m - 1.0)


def enumerate_candidate_sets() -> List[Tuple[int, ...]]:
    """
    返回候选下注集合列表。
    默认使用 config.DOUBLE_CANDIDATES：[(0,1),(1,2),...,(6,7)]
    """
    cands = getattr(config, "DOUBLE_CANDIDATES", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    return [tuple(map(int, t)) for t in cands]


def pick_best_set_for_match(match: Dict[str, Any], scheme: str) -> Dict[str, Any]:
    """
    给一场比赛选择最优下注集合，并写入：
      selected_set, p_selected, ev_selected, score, scheme, ev_fallback
    """
    rec = dict(match)  # shallow copy

    prob_dist = rec.get("prob_dist")
    odds_raw = rec.get("odds")
    odds = _to_int_key_odds(odds_raw)

    candidates = enumerate_candidate_sets()
    ev_min_p = float(getattr(config, "EV_MIN_P", 0.30))

    best_set_p: Optional[Tuple[int, ...]] = None
    best_p = -1.0
    best_ev_for_best_p = -math.inf

    best_set_ev: Optional[Tuple[int, ...]] = None
    best_ev = -math.inf
    best_p_for_best_ev = -1.0

    for s in candidates:
        p_s = compute_p_selected(prob_dist, s)
        ev_s = compute_ev(prob_dist, odds, s)

        # 赔率缺失会导致 ev_s = -inf，跳过
        if not math.isfinite(ev_s):
            continue

        # P 最优
        if p_s > best_p:
            best_p = p_s
            best_set_p = s
            best_ev_for_best_p = ev_s

        # EV 最优（带 p 下限）
        if p_s >= ev_min_p and ev_s > best_ev:
            best_ev = ev_s
            best_set_ev = s
            best_p_for_best_ev = p_s

    # 如果没有任何可下注集合（比如赔率全缺），保留占位，score 给极低
    if best_set_p is None and best_set_ev is None:
        rec["selected_set"] = None
        rec["p_selected"] = 0.0
        rec["ev_selected"] = -math.inf
        rec["score"] = -math.inf
        rec["scheme"] = scheme
        rec["ev_fallback"] = 0
        return rec

    scheme = (scheme or "P").upper()
    ev_fallback = 0

    if scheme == "EV":
        if best_set_ev is None:
            # EV 方案回退到 P
            ev_fallback = 1
            rec["selected_set"] = best_set_p
            rec["p_selected"] = float(best_p)
            rec["ev_selected"] = float(best_ev_for_best_p)
            rec["score"] = float(best_p)  # 回退时按 P 口径打分
        else:
            rec["selected_set"] = best_set_ev
            rec["p_selected"] = float(best_p_for_best_ev)
            rec["ev_selected"] = float(best_ev)
            rec["score"] = float(best_ev)
    else:
        # P 方案
        rec["selected_set"] = best_set_p
        rec["p_selected"] = float(best_p)
        rec["ev_selected"] = float(best_ev_for_best_p)
        rec["score"] = float(best_p)

    rec["scheme"] = scheme
    rec["ev_fallback"] = ev_fallback
    return rec


def select_matches_for_month(
    matches: List[Dict[str, Any]],
    min_bets: int,
    scheme: str,
    min_league_hist: int = 200,
) -> List[Dict[str, Any]]:
    """
    月度选场：
    1) 先给每场选最优集合 + 打分
    2) 优先选联赛历史样本足够（>=min_league_hist）的场次
    3) 若不足 min_bets，则从低样本联赛按 score 回填补齐
    """
    if not matches:
        return []

    scored = [pick_best_set_for_match(m, scheme) for m in matches]

    # 去掉“完全无赔率无法下注”的场次（score=-inf）
    scored = [s for s in scored if s.get("selected_set") is not None and math.isfinite(float(s.get("score", -math.inf)))]

    # 如果总可用都不足 min_bets，直接全返
    scored.sort(key=lambda x: (float(x.get("score", -math.inf)), x.get("date")), reverse=True)
    if len(scored) <= min_bets:
        # 标注字段，便于分析
        for s in scored:
            hist = int(s.get("league_hist_count", 0) or 0)
            s["eligible_league"] = 1 if hist >= min_league_hist else 0
            s["filled_from_low_hist"] = 0
        return scored

    # 联赛历史门槛
    eligible: List[Dict[str, Any]] = []
    low_hist: List[Dict[str, Any]] = []
    for s in scored:
        hist = int(s.get("league_hist_count", 0) or 0)
        if hist >= min_league_hist:
            eligible.append(s)
        else:
            low_hist.append(s)

    # 先取 eligible
    chosen: List[Dict[str, Any]] = []
    if len(eligible) >= min_bets:
        chosen = eligible[:min_bets]
        for s in chosen:
            s["eligible_league"] = 1
            s["filled_from_low_hist"] = 0
        return chosen

    # eligible 不足：全选 eligible + 从 low_hist 回填
    chosen = eligible[:]
    need = min_bets - len(chosen)
    filler = low_hist[:need]
    for s in chosen:
        s["eligible_league"] = 1
        s["filled_from_low_hist"] = 0
    for s in filler:
        s["eligible_league"] = 0
        s["filled_from_low_hist"] = 1
    return chosen + filler
