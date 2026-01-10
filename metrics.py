import pandas as pd


def _format_losing_streak_distribution(counts: dict) -> str:
    """把 {长度: 段数} 格式化为多行中文，便于写入 summary。"""
    if not counts:
        return "\n  无"
    lines = []
    for k in sorted(counts.keys()):
        lines.append(f"  连续{k}次未命中：{counts[k]} 段")
    return "\n" + "\n".join(lines)

def compute_metrics(df: pd.DataFrame) -> dict:
    """计算回测结果的核心指标（用于 summary.txt / compare.txt）。"""
    if df is None or len(df) == 0:
        return {
            "total_bets": 0,
            "hit_rate": 0.0,
            "max_losing_streak": 0,
            "losing_streak_distribution": "\n  无",
            "betting_days": 0,
            "natural_days": 0,
            "final_profit": 0.0,
            "final_bankroll": 0.0,
            "stoploss_triggers": 0,
        }

    # 命中/下注
    n_bets = int(len(df))
    n_hits = int(pd.to_numeric(df["hit"], errors="coerce").fillna(0).astype(int).sum())
    hit_rate = n_hits / n_bets if n_bets else 0.0

    # 止损次数
    stoploss_series = pd.to_numeric(df.get("stoploss", 0), errors="coerce").fillna(0).astype(int)
    stoploss_triggers = int(stoploss_series.sum())

    # 连续未命中段统计（按策略轮次：命中或止损会结束一段）
    hits = pd.to_numeric(df["hit"], errors="coerce").fillna(0).astype(int).tolist()
    stoplosses = stoploss_series.tolist()

    counts = {}  # {连续未命中长度: 段数}
    cur = 0
    max_losing_streak = 0
    for h, sl in zip(hits, stoplosses):
        if h == 0:
            cur += 1
            if cur > max_losing_streak:
                max_losing_streak = cur
            # 止损意味着“本轮结束”，把这一段记下来并重置
            if sl == 1:
                counts[cur] = counts.get(cur, 0) + 1
                cur = 0
        else:
            # 命中：如果之前有一段未命中，先结算那一段
            if cur > 0:
                counts[cur] = counts.get(cur, 0) + 1
                cur = 0

    # 收尾：最后如果还在未命中段里，也要记一次
    if cur > 0:
        counts[cur] = counts.get(cur, 0) + 1

    losing_streak_distribution = _format_losing_streak_distribution(counts)

    # 收益
    total_profit = float(pd.to_numeric(df["profit"], errors="coerce").fillna(0).sum())
    final_bankroll = float(pd.to_numeric(df["bankroll"], errors="coerce").iloc[-1])

    # 时间维度（尽量兼容 date 为字符串的情况）
    date_series = pd.to_datetime(df["date"], errors="coerce")
    betting_days = int(date_series.nunique())
    if betting_days == 0:
        natural_days = 0
    else:
        natural_days = int((date_series.max() - date_series.min()).days + 1)

    return {
        "total_bets": n_bets,
        "hit_rate": round(hit_rate, 4),
        "max_losing_streak": int(max_losing_streak),
        "losing_streak_distribution": losing_streak_distribution,
        "betting_days": betting_days,
        "natural_days": natural_days,
        "final_profit": round(total_profit, 2),
        "final_bankroll": round(final_bankroll, 2),
        "stoploss_triggers": stoploss_triggers,
    }
