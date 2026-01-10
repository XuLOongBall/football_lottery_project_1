import pandas as pd

RESULT_PATH = "outputs/best_results.csv"  # 改成实际文件名（英文或中文都行）

df = pd.read_csv(RESULT_PATH)
print("columns:", df.columns.tolist())

# 自动兼容：英文/中文列名
col_month = "month_id" if "month_id" in df.columns else ("月份" if "月份" in df.columns else None)
col_void  = "is_void" if "is_void" in df.columns else None
col_profit = "profit" if "profit" in df.columns else ("本场收益" if "本场收益" in df.columns else None)
col_hit = "hit" if "hit" in df.columns else ("是否命中" if "是否命中" in df.columns else None)

if col_month is None:
    raise KeyError("找不到 month_id/月份 列，无法做按月统计。")

m = df.groupby(col_month).size()
print("min bets per month:", int(m.min()))
print("total bets:", len(df))

if col_void is not None:
    void_cnt = int((df[col_void] == 1).sum())
    print("void count:", void_cnt)

if col_profit is not None and col_void is not None:
    print("void profit unique:", df.loc[df[col_void] == 1, col_profit].unique()[:10])

# 命中率（剔除 void）
if col_hit is not None:
    if col_hit == "hit":
        decided = df[(df[col_void] != 1) & (df[col_hit].isin([0, 1]))]
        hit_rate = decided[col_hit].mean() if len(decided) else 0.0
    else:
        decided = df[(df[col_void] != 1) & (df[col_hit].isin(["命中", "未命中"]))]
        hit_rate = (decided[col_hit] == "命中").mean() if len(decided) else 0.0

    print("decided bets:", len(decided))
    print("hit rate (excluding void):", hit_rate)

# 最终累计收益
final_profit = df["累计收益"].iloc[-1]
print("final profit:", final_profit)

# 最大回撤（用累计收益当资金曲线）
equity = df["累计收益"].astype(float)
peak = equity.cummax()
dd = equity - peak
max_dd = dd.min()
print("max drawdown:", max_dd)

