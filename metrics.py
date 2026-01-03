def compute_metrics(df):
    n_bets = len(df)
    n_hits = df["hit"].sum()
    hit_rate = n_hits / n_bets

    # 最大连续不中（按策略轮次：命中或止损会重置）
    max_losing_streak = 0
    cur = 0
    for h, sl in zip(df["hit"], df.get("stoploss", 0)):
       if h == 0:
           cur += 1
           max_losing_streak = max(max_losing_streak, cur)
           if sl == 1:   # 止损：本轮结束
            cur = 0
       else:
           cur = 0

    total_profit = df["profit"].sum()
    final_bankroll = df["bankroll"].iloc[-1]

    betting_days = df["date"].nunique()
    natural_days = (df["date"].max() - df["date"].min()).days + 1
    
    stoploss_triggers = int(df.get("stoploss", 0).sum())

    return {
        "total_bets": n_bets,
        "hit_rate": round(hit_rate, 4),
        "max_losing_streak": max_losing_streak,
        "betting_days": betting_days,
        "natural_days": natural_days,
        "final_profit": round(total_profit, 2),
        "final_bankroll": round(final_bankroll, 2),
        "stoploss_triggers": stoploss_triggers,
    }
