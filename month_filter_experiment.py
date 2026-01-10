"""
month_filter_experiment.py

用途：
- 做“月份级别”的下注可行性/限制条件试验（不泄露未来）
- 帮你回答：在不同限制下，哪些月份会被 backtest 跳过？是否还能保证每月>=15、总>=1800？

核心原则（避免泄露未来）：
- 对每个月 m，只允许使用 month_start 之前的数据作为历史（date < month_start）
- 若设置了 lookback_days，则历史再额外限制为 [month_start - lookback_days, month_start)

输出：
- outputs/month_report.csv：每个月的比赛数、历史样本量（不同lookback）
- outputs/grid_summary.csv：不同(lookback_days, rolling_window)组合下的“会跳过月份数/总下注数”等汇总
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Dict, Tuple

import numpy as np
import pandas as pd

# 复用项目里的读取/清洗（若失败会回退到 pandas 读取）
try:
    import config
except Exception:
    config = None

try:
    from io_loader import load_raw_data
    from cleaning import clean_data
except Exception:
    load_raw_data = None
    clean_data = None


# -----------------------------
# Helpers
# -----------------------------

def _parse_int_list(x: str) -> List[int]:
    """
    支持：
    - "180,365,730"
    - "180 365 730"
    - "none,365"（none 会被忽略；请用 --include-all-history 来包含“无限历史”版本）
    """
    if x is None or str(x).strip() == "":
        return []
    parts = [p.strip().lower() for p in str(x).replace(",", " ").split()]
    out = []
    for p in parts:
        if p in ("none", "null"):
            continue
        out.append(int(p))
    return out


def _detect_league_col(df: pd.DataFrame) -> Optional[str]:
    """
    尝试从原始/清洗后的 df 中找“联赛/赛事名称”列。

    注意：很多数据把“对阵(A VS B)”放在“赛事/主客场/对阵”等列里，
    这些列名容易误判成“联赛”。这里优先选择更明确的列名，并做一次内容自检。
    """
    # 越靠前优先级越高
    cands = [
        "league", "联赛", "联赛名称", "赛事名称", "League", "competition", "Competition"
    ]
    for c in cands:
        if c in df.columns:
            s = df[c].astype(str)
            # 如果大量值看起来像“对阵”，则不当作联赛列
            vs_like = s.str.contains(r"\bvs\b|\bVS\b|\s-\s|—|–|：|:", case=False, regex=True).mean()
            if vs_like >= 0.30:
                continue
            return c
    return None


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_and_clean(data_file: Optional[str]) -> pd.DataFrame:
    """
    读取 + 清洗。优先走项目的 io_loader/cleaning（编码/字段更稳），否则直接 pandas.read_csv。
    """
    # 1) 走项目函数
    if load_raw_data is not None and clean_data is not None:
        if data_file is not None:
            # 临时覆盖 config.DATA_FILE（只影响当前进程）
            if config is not None and hasattr(config, "DATA_FILE"):
                setattr(config, "DATA_FILE", data_file)
        df_raw = load_raw_data()
        df, log = clean_data(df_raw)
        return df

    # 2) 回退
    if data_file is None:
        raise FileNotFoundError("没有提供 data_file，且无法导入项目的 io_loader/config。请用 --data-file 指定 CSV 路径。")
    try:
        df_raw = pd.read_csv(data_file, encoding="gbk", low_memory=False, encoding_errors="replace")
    except Exception:
        df_raw = pd.read_csv(data_file, encoding="utf-8", low_memory=False)
    # 尝试最小清洗：确保 date/month_id/total_goals 存在
    df = df_raw.copy()
    if "date" not in df.columns:
        # 常见列名
        for c in ["比赛时间", "时间", "开赛时间"]:
            if c in df.columns:
                df["date"] = pd.to_datetime(df[c], errors="coerce")
                break
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if "month_id" not in df.columns:
        df["month_id"] = df["date"].dt.strftime("%Y-%m")
    if "total_goals" not in df.columns:
        if "总进球" in df.columns:
            df["total_goals"] = pd.to_numeric(df["总进球"], errors="coerce")
        else:
            raise ValueError("回退清洗失败：找不到 total_goals/总进球 列。建议使用项目自带 cleaning.clean_data。")
    df = df.dropna(subset=["total_goals"])
    df["total_goals"] = df["total_goals"].astype(int)
    df.loc[df["total_goals"] >= 7, "total_goals"] = 7
    return df



def _build_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month_id")
    out = pd.DataFrame({
        "month_id": g.size().index,
        "n_matches": g.size().values,
        "month_start": g["date"].min().values,
        "month_end": g["date"].max().values,
    })
    out = out.sort_values("month_start").reset_index(drop=True)
    return out


def _history_count_by_month(dates_sorted: np.ndarray,
                           month_starts: np.ndarray,
                           lookback_days: Optional[int]) -> np.ndarray:
    """
    dates_sorted: 全体比赛日期（升序，datetime64[ns]）
    month_starts: 每个月第一场日期（datetime64[ns]）
    lookback_days: None 表示不限；否则只允许使用 [start - lookback_days, start)
    返回：每个月可用历史样本数（严格 date < month_start）
    """
    start_idx = np.searchsorted(dates_sorted, month_starts, side="left")
    if lookback_days is None:
        left_idx = np.zeros_like(start_idx)
    else:
        left_bound = month_starts - np.timedelta64(int(lookback_days), "D")
        left_idx = np.searchsorted(dates_sorted, left_bound, side="left")
    return (start_idx - left_idx).astype(int)


def _simulate_total_bets(month_df: pd.DataFrame,
                        hist_count: np.ndarray,
                        monthly_min_bets: int,
                        rolling_window: int,
                        strict_skip_if_hist_lt_window: bool) -> Tuple[int, int]:
    """
    返回：(total_bets, skipped_months_count)
    strict_skip_if_hist_lt_window=True 模拟你现在 backtest.py 的 continue 行为
    """
    n_month = month_df["n_matches"].values.astype(int)
    if strict_skip_if_hist_lt_window:
        skip = hist_count < int(rolling_window)
        bets = np.where(skip, 0, np.minimum(n_month, int(monthly_min_bets)))
        return int(bets.sum()), int(skip.sum())
    else:
        bets = np.minimum(n_month, int(monthly_min_bets))
        return int(bets.sum()), 0


def _league_low_history_pct(df: pd.DataFrame,
                            month_ids: np.ndarray,
                            month_starts: np.ndarray,
                            lookback_days: Optional[int],
                            min_league_history: int,
                            league_col: str) -> np.ndarray:
    """
    计算每个月：当月比赛中，有多少比例的比赛所属联赛在历史窗口内样本量 < min_league_history。
    仅用于“风险提示”，不用于跳月。
    """
    # 预处理：每个联赛的日期数组（升序）
    league_dates: Dict[str, np.ndarray] = {}
    for L, sub in df[[league_col, "date"]].dropna().groupby(league_col):
        league_dates[str(L)] = np.sort(sub["date"].values.astype("datetime64[ns]"))

    # 逐月统计（月份只有 ~120，速度足够）
    pct_list = []
    for m, start in zip(month_ids, month_starts):
        df_month = df[df["month_id"] == m]
        if df_month.empty:
            pct_list.append(0.0)
            continue
        # 各联赛在当月出现的次数
        vc = df_month[league_col].astype(str).value_counts(dropna=True)
        bad = 0
        total = int(vc.sum())
        for L, cnt in vc.items():
            d = league_dates.get(L)
            if d is None or d.size == 0:
                bad += int(cnt)
                continue
            # 计算该联赛在窗口内历史数（严格 date < start）
            s_idx = np.searchsorted(d, start, side="left")
            if lookback_days is None:
                l_idx = 0
            else:
                lb = start - np.timedelta64(int(lookback_days), "D")
                l_idx = np.searchsorted(d, lb, side="left")
            n_hist = int(s_idx - l_idx)
            if n_hist < int(min_league_history):
                bad += int(cnt)
        pct_list.append(bad / total if total > 0 else 0.0)
    return np.array(pct_list, dtype=float)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default=(getattr(config, "DATA_FILE", None) if config else None),
                        help="原始CSV路径（默认取 config.DATA_FILE）")
    parser.add_argument("--out-dir", type=str, default=(getattr(config, "OUTPUT_DIR", "outputs") if config else "outputs"),
                        help="输出目录（默认 outputs）")

    parser.add_argument("--monthly-min-bets", type=int, default=(getattr(config, "MONTHLY_MIN_BETS", 15) if config else 15),
                        help="每月最少下注场数（硬约束）")
    parser.add_argument("--total-min-bets", type=int, default=(getattr(config, "TOTAL_MIN_BETS", 1800) if config else 1800),
                        help="总下注场数下限（硬约束）")

    parser.add_argument("--rolling-windows", type=str, default=str(getattr(config, "ROLLING_WINDOW", 200) if config else 200),
                        help="用于“严格跳月”模拟的 rolling_window 列表，如 '50,100,200'")
    parser.add_argument("--lookback-days", type=str, default="365,730",
                        help="限制历史不过久远：只允许用最近X天历史。支持列表，如 '180,365,730'")
    parser.add_argument("--include-all-history", action="store_true",
                        help="同时包含 lookback_days=None 的版本（不限历史）")

    parser.add_argument("--league-min-history", type=int, default=50,
                        help="联赛历史样本阈值（用于风险提示：当月比赛中联赛历史不足的占比）")

    args = parser.parse_args()

    _safe_mkdir(args.out_dir)

    df = _read_and_clean(args.data_file)
    df = df.sort_values("date").reset_index(drop=True)

    # 基础 month stats
    month_stats = _build_month_stats(df)

    # 统一日期数组（全体）
    dates_sorted = df["date"].values.astype("datetime64[ns]")
    month_starts = month_stats["month_start"].values.astype("datetime64[ns]")

    # 解析参数列表
    rolling_windows = _parse_int_list(args.rolling_windows) or [int(getattr(config, "ROLLING_WINDOW", 200) if config else 200)]
    lookbacks = _parse_int_list(args.lookback_days)
    lookback_opts: List[Optional[int]] = []
    if args.include_all_history:
        lookback_opts.append(None)
    lookback_opts += lookbacks

    # month_report：每个 lookback 都给出 hist_count
    report = month_stats.copy()
    for H in lookback_opts:
        col = f"hist_count_{'ALL' if H is None else H}d"
        report[col] = _history_count_by_month(dates_sorted, month_starts, H)

    # 联赛风险占比（可选）
    league_col = _detect_league_col(df)
    if league_col is not None:
        for H in lookback_opts:
            col = f"pct_league_hist_lt_{args.league_min_history}_{'ALL' if H is None else H}d"
            report[col] = _league_low_history_pct(
                df=df,
                month_ids=month_stats["month_id"].values,
                month_starts=month_starts,
                lookback_days=H,
                min_league_history=args.league_min_history,
                league_col=league_col
            )
        report["league_col_used"] = league_col
    else:
        report["league_col_used"] = ""

    # 标记：当月清洗后场次是否不足 monthly_min_bets（会影响总下注>=1800）
    report["month_matches_lt_min_bets"] = report["n_matches"] < int(args.monthly_min_bets)

    # 写 month_report
    month_report_path = os.path.join(args.out_dir, "month_report.csv")
    report.to_csv(month_report_path, index=False, encoding="utf-8-sig")

    # （可选）输出联赛总体统计，便于你确认“联赛列到底是哪一列、有哪些联赛、样本量多大”
    if league_col is not None:
        league_summary = (
            df[[league_col, "date"]]
            .dropna()
            .assign(**{league_col: lambda x: x[league_col].astype(str).str.strip()})
            .groupby(league_col)
            .agg(n=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
            .sort_values("n", ascending=False)
            .reset_index()
        )
        league_summary_path = os.path.join(args.out_dir, "league_summary.csv")
        league_summary.to_csv(league_summary_path, index=False, encoding="utf-8-sig")
    else:
        league_summary_path = ""

    # grid_summary：不同 (lookback, rolling_window) 下，模拟“严格跳月”会跳多少月、总下注是多少
    rows = []
    for H in lookback_opts:
        hist_count = _history_count_by_month(dates_sorted, month_starts, H)

        # 不跳月（建议的做法）：总下注由 min(month_matches, monthly_min_bets) 决定
        total_no_skip, _ = _simulate_total_bets(
            month_df=month_stats,
            hist_count=hist_count,
            monthly_min_bets=int(args.monthly_min_bets),
            rolling_window=int(rolling_windows[0]),
            strict_skip_if_hist_lt_window=False
        )

        for W in rolling_windows:
            total_strict, skipped = _simulate_total_bets(
                month_df=month_stats,
                hist_count=hist_count,
                monthly_min_bets=int(args.monthly_min_bets),
                rolling_window=int(W),
                strict_skip_if_hist_lt_window=True
            )

            rows.append({
                "lookback_days": ("ALL" if H is None else int(H)),
                "rolling_window": int(W),
                "months_total": int(len(month_stats)),
                "months_skipped_if_strict": int(skipped),
                "total_bets_if_strict": int(total_strict),
                "total_bets_if_no_skip": int(total_no_skip),
                "months_with_lt_min_bets": int(report["month_matches_lt_min_bets"].sum()),
                "min_hist_count": int(hist_count.min()),
                "p10_hist_count": int(np.quantile(hist_count, 0.10)),
                "median_hist_count": int(np.median(hist_count)),
                "p90_hist_count": int(np.quantile(hist_count, 0.90)),
                "meets_total_min_bets_if_strict": bool(total_strict >= int(args.total_min_bets)),
                "meets_total_min_bets_if_no_skip": bool(total_no_skip >= int(args.total_min_bets)),
            })

    grid = pd.DataFrame(rows).sort_values(["lookback_days", "rolling_window"]).reset_index(drop=True)
    grid_path = os.path.join(args.out_dir, "grid_summary.csv")
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")

    # 控制台输出：给你一个快速结论
    print(f"[OK] month_report.csv -> {month_report_path}")
    print(f"[OK] grid_summary.csv -> {grid_path}")
    if league_summary_path:
        print(f"[OK] league_summary.csv -> {league_summary_path}")

    # 给出“严格跳月”为0的组合（如果存在）
    ok0 = grid[grid["months_skipped_if_strict"] == 0]
    if len(ok0) > 0:
        best = ok0.sort_values(["lookback_days", "rolling_window"]).head(10)
        print("\n组合筛选：months_skipped_if_strict == 0 的前10条：")
        print(best[["lookback_days","rolling_window","total_bets_if_strict","min_hist_count","p10_hist_count","median_hist_count"]].to_string(index=False))
    else:
        print("\n提示：在你给的 lookback_days / rolling_window 组合里，没有任何组合能做到 strict 跳月为0。")
        print("如果你必须保证每月都有下注（总>=1800），建议移除 backtest 里的 'if len(history)<ROLLING_WINDOW: continue'，改成历史不足就用已有历史/冷启动。")

if __name__ == "__main__":
    main()
