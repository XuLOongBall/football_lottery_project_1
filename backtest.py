import pandas as pd

import config
from model.poisson_model import predict_goal_distribution
from selection import select_matches_for_month
from betting import settle_bet


def run_backtest(df: pd.DataFrame, scheme: str = "P"):
    """
    主回测流程：按月选场 + 按时间下注
    scheme="EV" -> 方案A（按EV排序选场）
    scheme="P"  -> 方案B（按p_S排序选场）
    """
    results = []

    martingale_level = 0
    bankroll = 0.0

    df = df.sort_values("date").reset_index(drop=True)

    for month, df_month in df.groupby("month_id"):
        df_history = df[df["date"] < df_month["date"].min()]

        if len(df_history) < config.ROLLING_WINDOW:
            continue

        month_matches = []
        history_goals = df_history["total_goals"].values[-config.ROLLING_WINDOW:]

        for _, row in df_month.iterrows():
            prob_dist = predict_goal_distribution(history_goals)

            match_record = {
                "date": row["date"],
                "month_id": row["month_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "y_true": row["total_goals"],
                "odds": row["odds"],
                "prob_dist": prob_dist
            }
            month_matches.append(match_record)

        # ✅ 这里：把 scheme 传进去
        selected_matches = select_matches_for_month(
            month_matches,
            min_bets=config.MONTHLY_MIN_BETS,
            scheme=scheme
        )
        if not selected_matches:
            continue

        selected_matches.sort(key=lambda x: x["date"])

        for match in selected_matches:
            settle = settle_bet(
                selected_set=match["selected_set"],
                odds=match["odds"],
                y_true=match["y_true"],
                level=martingale_level
            )

            martingale_level = settle["next_level"]
            bankroll += settle["profit"]

            result = {
                "date": match["date"],
                "month_id": match["month_id"],
                "scheme": scheme,                         # ✅ 新增
                "ev_fallback": match.get("ev_fallback", 0),# ✅ 新增
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "y_true": match["y_true"],
                "selected_set": match["selected_set"],
                "p_selected": match["p_selected"],
                "ev_selected": match.get("ev_selected"),  # ✅ 新增
                "score": match.get("score"),              # ✅ 新增（排序依据）
                "stake_total": settle["stake_total"],
                "stake_split": settle["stake_split"],
                "hit": settle["hit"],
                "profit": settle["profit"],
                "stoploss": settle["stoploss"],
                "bankroll": bankroll,
                "martingale_level": martingale_level

            }

            results.append(result)

    return results
