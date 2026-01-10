import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _set_chinese_font():
    """尽量设置一个可用的中文字体（不同系统字体名不同）。"""
    candidates = [
        "Microsoft YaHei",   # Windows
        "SimHei",            # Windows
        "PingFang SC",       # macOS
        "Heiti SC",          # macOS
        "Noto Sans CJK SC",  # Linux (可能)
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


_CN_FONT = _set_chinese_font()


def _save_both(fig_path_en: str, fig_path_cn: str, dpi: int = 200):
    """同时保存英文文件名和中文文件名，避免你后续脚本依赖旧名字。"""
    plt.savefig(fig_path_en, dpi=dpi)
    if fig_path_cn and fig_path_cn != fig_path_en:
        try:
            plt.savefig(fig_path_cn, dpi=dpi)
        except Exception:
            pass


def plot_equity_curve(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(df["bankroll"])
    plt.xlabel("下注序号")
    plt.ylabel("累计收益")
    plt.title("资金曲线")
    plt.tight_layout()

    p_en = os.path.join(output_dir, "equity_curve.png")
    p_cn = os.path.join(output_dir, "资金曲线.png")
    _save_both(p_en, p_cn, dpi=150)
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
        x_label = "日期"
    else:
        x = np.arange(len(df))
        x_label = "序号"

    equity = df["bankroll"].astype(float).values
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak  # <= 0

    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(x, drawdown)
    plt.title("回撤曲线")
    plt.xlabel(x_label)
    plt.ylabel("回撤（累计收益-历史峰值）")
    plt.tight_layout()

    p_en = os.path.join(output_dir, "drawdown_curve.png")
    p_cn = os.path.join(output_dir, "回撤曲线.png")
    _save_both(p_en, p_cn, dpi=200)
    plt.close()


def plot_monthly_stats(df_results: pd.DataFrame, output_dir: str):
    """
    月度下注次数 & 月度命中率（分成两张图，更清晰）
    - 改进：自动“抽稀”横轴刻度（最多显示约 12 个），避免月份太多挤成一坨
    """
    if df_results.empty:
        return

    df = df_results.copy()
    if "month_id" not in df.columns:
        return

    # hit 允许 NaN（例如 void 退票），NaN 不计入命中率/连错段
    df["hit_num"] = pd.to_numeric(df.get("hit"), errors="coerce")
    df["decided"] = df["hit_num"].notna().astype(int)

    grp = df.groupby("month_id", sort=True).agg(
        bets=("decided", "size"),          # 含 void 的总下注笔数
        decided_bets=("decided", "sum"),   # 已结算笔数（不含 void）
        hit_rate=("hit_num", "mean"),      # 仅在 decided 上计算（pandas mean 默认忽略 NaN）
    ).reset_index()

    labels = grp["month_id"].astype(str).tolist()
    x = np.arange(len(labels))

    # 横轴最多显示 ~12 个刻度（月份多时自动抽稀）
    max_ticks = 12
    step = max(1, int(np.ceil(len(labels) / max_ticks)))
    tick_idx = x[::step]
    tick_labels = [labels[i] for i in tick_idx]

    os.makedirs(output_dir, exist_ok=True)

    # 1) 月度下注次数
    plt.figure(figsize=(10, 4))
    plt.plot(x, grp["bets"].astype(int).values)
    plt.title("月度下注次数")
    plt.xlabel("月份")
    plt.ylabel("下注次数")
    plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    p_en = os.path.join(output_dir, "monthly_bets.png")
    p_cn = os.path.join(output_dir, "月度下注次数.png")
    _save_both(p_en, p_cn, dpi=200)
    plt.close()

    # 2) 月度命中率
    plt.figure(figsize=(10, 4))
    plt.plot(x, grp["hit_rate"].astype(float).values)
    plt.title("月度命中率")
    plt.xlabel("月份")
    plt.ylabel("命中率")
    plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    p_en = os.path.join(output_dir, "monthly_hit_rate.png")
    p_cn = os.path.join(output_dir, "月度命中率.png")
    _save_both(p_en, p_cn, dpi=200)
    plt.close()
