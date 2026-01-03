import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_equity_curve(df, output_dir):
    plt.figure()
    plt.plot(df["bankroll"])
    plt.xlabel("Bet index")
    plt.ylabel("Cumulative profit")
    plt.title("Equity Curve")

    path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
 


def plot_drawdown_curve(df_results: pd.DataFrame, output_dir: str):
    """
    最大回撤曲线：DD_t = bankroll_t - cummax(bankroll_t)
    画出来是 <= 0 的曲线（越往下回撤越大）
    """
    if df_results.empty:
        return

    df = df_results.copy()

    # 处理时间轴
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        x = df["date"]
    else:
        x = np.arange(len(df))

    equity = df["bankroll"].astype(float).values
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak  # <= 0

    plt.figure()
    plt.plot(x, drawdown)
    plt.title("Drawdown Curve")
    plt.xlabel("Date" if "date" in df.columns else "Index")
    plt.ylabel("Drawdown (bankroll - peak)")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "drawdown_curve.png"), dpi=200)
    plt.close()


def plot_monthly_stats(df_results: pd.DataFrame, output_dir: str):
    """
    月度交易数 & 月度命中率（分成两张图，更清晰）
    """
    if df_results.empty:
        return

    df = df_results.copy()
    if "month_id" not in df.columns:
        return

    grp = df.groupby("month_id", sort=True).agg(
        bets=("hit", "size"),
        hit_rate=("hit", "mean"),
    ).reset_index()

    # 1) 月度交易数
    plt.figure()
    plt.plot(grp["month_id"].astype(str), grp["bets"].astype(int))
    plt.title("Monthly Number of Bets")
    plt.xlabel("Month")
    plt.ylabel("Bets")
    plt.xticks(rotation=60)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "monthly_bets.png"), dpi=200)
    plt.close()

    # 2) 月度命中率
    plt.figure()
    plt.plot(grp["month_id"].astype(str), grp["hit_rate"].astype(float))
    plt.title("Monthly Hit Rate")
    plt.xlabel("Month")
    plt.ylabel("Hit rate")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_hit_rate.png"), dpi=200)
    plt.close()
