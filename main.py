import os
import pandas as pd
import shutil

import config
from io_loader import load_raw_data
from cleaning import clean_data
from backtest import run_backtest
from plotting import plot_equity_curve, plot_drawdown_curve, plot_monthly_stats
from metrics import compute_metrics


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _write_kv(f, label: str, value):
    """写一条指标到文本文件：支持多行排版。"""
    if value is None:
        f.write(f"{label}: None\n")
        return

    # dict：逐行展开
    if isinstance(value, dict):
        f.write(f"{label}:\n")
        for k in sorted(value.keys()):
            f.write(f"  {k}: {value[k]}\n")
        return

    s = str(value)
    if "\n" in s:
        # 如果 value 本身以换行开头，直接接在冒号后面更好看
        if s.startswith("\n"):
            f.write(f"{label}:{s}\n")
        else:
            f.write(f"{label}:\n{s}\n")
    else:
        f.write(f"{label}: {s}\n")



def export_best_scheme_to_root(best_scheme: str):
    """
    把 outputs/EV 或 outputs/P 里的一套结果，复制一份到 outputs/ 根目录。
    文件名会加 best_ 前缀，避免覆盖 compare.txt 等文件。
    """
    src_dir = os.path.join(config.OUTPUT_DIR, best_scheme)
    dst_dir = config.OUTPUT_DIR
    ensure_dir(dst_dir)

    # 记录“最佳方案是谁”
    best_flag_path = os.path.join(dst_dir, "best_scheme.txt")
    with open(best_flag_path, "w", encoding="utf-8") as f:
        f.write(f"best_scheme={best_scheme}\n")
        f.write(f"best_scheme_cn={SCHEME_NAME_CN.get(best_scheme, best_scheme)}\n")

    # 想复制到根目录的文件清单（存在就复制，不存在就跳过）
    copy_list = [
        # 逐场结果
        ("results.csv", "best_results.csv"),
        ("results_en.csv", "best_results_en.csv"),
        # 汇总
        ("summary.txt", "best_summary.txt"),
        ("summary_en.txt", "best_summary_en.txt"),
        # 图（英文名）
        ("equity_curve.png", "best_equity_curve.png"),
        ("drawdown_curve.png", "best_drawdown_curve.png"),
        ("monthly_bets.png", "best_monthly_bets.png"),
        ("monthly_hit_rate.png", "best_monthly_hit_rate.png"),
        # 图（中文名，若 plotting.py 同时输出）
        ("资金曲线.png", "best_资金曲线.png"),
        ("回撤曲线.png", "best_回撤曲线.png"),
        ("月度下注次数.png", "best_月度下注次数.png"),
        ("月度命中率.png", "best_月度命中率.png"),
    ]

    for src_name, dst_name in copy_list:
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(dst_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)



# =========================
# 中文输出映射（只影响“写出来的结果”，不影响回测内部计算）
# =========================
SCHEME_NAME_CN = {
    "EV": "方案A（EV：按期望收益选）",
    "P":  "方案B（P：按命中概率选）",
}

METRIC_LABEL_CN = {
    "total_bets": "总下注次数",
    "hit_rate": "命中率",
    "max_losing_streak": "最大连续未命中次数",
    "losing_streak_distribution": "连续未命中段统计（长度→段数）",
    "betting_days": "实际下注天数",
    "natural_days": "覆盖自然天数",
    "final_profit": "最终总收益",
    "final_bankroll": "最终累计收益",
    "stoploss_triggers": "触发止损次数",
    "max_drawdown": "最大回撤（金额）",
}

RESULT_COL_CN = {
    "date": "日期",
    "month_id": "月份",
    "scheme": "方案",
    "ev_fallback": "EV回退",
    "home_team": "主队",
    "away_team": "客队",
    "y_true": "真实总进球",
    "selected_set": "下注选项",
    "p_selected": "命中概率",
    "ev_selected": "期望收益EV",
    "score": "排序分数",
    "stake_total": "总下注额",
    "stake_split": "下注分配",
    "hit": "是否命中",
    "profit": "本场收益",
    "stoploss": "是否止损",
    "bankroll": "累计收益",
    "martingale_level": "当前倍投层级",
}


def _format_selected_set(x):
    # (2, 3) -> "2/3"
    try:
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            return "/".join(str(i) for i in x)
    except Exception:
        pass
    return x


def to_chinese_results(df_results: pd.DataFrame) -> pd.DataFrame:
    df = df_results.copy()

    # 值层面的中文化（更好读）
    if "scheme" in df.columns:
        df["scheme"] = df["scheme"].map(lambda s: SCHEME_NAME_CN.get(s, s))
    if "selected_set" in df.columns:
        df["selected_set"] = df["selected_set"].map(_format_selected_set)
    if "hit" in df.columns:
        df["hit"] = df["hit"].map(lambda v: "命中" if int(v) == 1 else "未命中")
    if "stoploss" in df.columns:
        df["stoploss"] = df["stoploss"].map(lambda v: "是" if int(v) == 1 else "否")
    if "ev_fallback" in df.columns:
        df["ev_fallback"] = df["ev_fallback"].map(lambda v: "是" if int(v) == 1 else "否")

    # 列名中文化
    df = df.rename(columns=RESULT_COL_CN)
    return df


def write_summary_cn(summary: dict, out_dir: str, scheme: str):
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"方案：{SCHEME_NAME_CN.get(scheme, scheme)}\n")
        f.write("-" * 30 + "\n")
        for k, v in summary.items():
            label = METRIC_LABEL_CN.get(k, k)
            _write_kv(f, label, v)


def run_one_scheme(df_clean: pd.DataFrame, scheme: str):
    """
    scheme="EV" -> 方案A（按EV选场）
    scheme="P"  -> 方案B（按p_S选场）
    """
    out_dir = os.path.join(config.OUTPUT_DIR, scheme)
    ensure_dir(out_dir)

    # ✅ 把全局清洗日志复制到该方案目录（可选但很推荐）
    src = os.path.join(config.OUTPUT_DIR, "cleaning_log.txt")
    dst = os.path.join(out_dir, "cleaning_log.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)

    # 回测
    results = run_backtest(df_clean, scheme=scheme)

    if len(results) == 0:
        raise RuntimeError(f"没有产生任何下注结果：scheme={scheme}")

    df_results = pd.DataFrame(results)

    # 1) 保存逐场结果（英文原始版，便于代码/调试）
    df_results.to_csv(
        os.path.join(out_dir, "results_en.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 2) 保存逐场结果（中文展示版：results.csv）
    df_results_cn = to_chinese_results(df_results)
    df_results_cn.to_csv(
        os.path.join(out_dir, "results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 汇总指标（内部仍用英文列名的 df_results）
    summary = compute_metrics(df_results)

    # 最大回撤（金额，正数表示最大回撤幅度）
    try:
        equity = df_results["bankroll"].astype(float)
        peak = equity.cummax()
        dd = equity - peak  # <= 0
        max_dd = float(dd.min()) if len(dd) else 0.0
        summary["max_drawdown"] = round(abs(max_dd), 2)
    except Exception:
        summary["max_drawdown"] = None


    # 写中文 summary.txt（同时保留一份英文 key 的 summary_en.txt）
    write_summary_cn(summary, out_dir=out_dir, scheme=scheme)
    with open(os.path.join(out_dir, "summary_en.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            _write_kv(f, k, v)

    # 画图（plotting.py 我们也会改成中文标题/坐标）
    plot_equity_curve(df_results, output_dir=out_dir)
    plot_drawdown_curve(df_results, output_dir=out_dir)
    plot_monthly_stats(df_results, output_dir=out_dir)

    return summary


def main():
    ensure_dir(config.OUTPUT_DIR)

    # 1. 读取原始数据
    df_raw = load_raw_data()

    # 2. 清洗数据
    df_clean, clean_log = clean_data(df_raw)

    # 保存清洗日志（全局一份就够）
    with open(os.path.join(config.OUTPUT_DIR, "cleaning_log.txt"), "w", encoding="utf-8") as f:
        for line in clean_log:
            f.write(line + "\n")

    # 3-6. 分别跑两套方案
    summary_ev = run_one_scheme(df_clean, scheme="EV")  # 方案A
    summary_p  = run_one_scheme(df_clean, scheme="P")   # 方案B

    # 7. 输出对比文件（方便写报告）
    compare_path = os.path.join(config.OUTPUT_DIR, "compare.txt")
    with open(compare_path, "w", encoding="utf-8") as f:
        f.write("=== 方案A（EV） ===\n")
        for k, v in summary_ev.items():
            _write_kv(f, METRIC_LABEL_CN.get(k, k), v)

        f.write("\n=== 方案B（P） ===\n")
        for k, v in summary_p.items():
            _write_kv(f, METRIC_LABEL_CN.get(k, k), v)


    # 8. 选“更优秀”的方案：先看实际最终收益；相同则看最大回撤；再看止损次数；再看最大连续未命中
    def _pick_best(s_ev: dict, s_p: dict) -> str:
        # 1) 最终收益（越大越好）
        ev_profit = s_ev.get("final_profit")
        p_profit  = s_p.get("final_profit")
        if ev_profit is not None and p_profit is not None and ev_profit != p_profit:
            return "EV" if ev_profit > p_profit else "P"

        # 2) 最大回撤（越小越好）
        ev_dd = s_ev.get("max_drawdown")
        p_dd  = s_p.get("max_drawdown")
        if ev_dd is not None and p_dd is not None and ev_dd != p_dd:
            return "EV" if ev_dd < p_dd else "P"

        # 3) 止损次数（越少越好）
        ev_sl = s_ev.get("stoploss_triggers")
        p_sl  = s_p.get("stoploss_triggers")
        if ev_sl is not None and p_sl is not None and ev_sl != p_sl:
            return "EV" if ev_sl < p_sl else "P"

        # 4) 最大连续未命中（越小越好）
        ev_ls = s_ev.get("max_losing_streak")
        p_ls  = s_p.get("max_losing_streak")
        if ev_ls is not None and p_ls is not None and ev_ls != p_ls:
            return "EV" if ev_ls < p_ls else "P"

        # 打平默认选 EV
        return "EV"

    best_scheme = _pick_best(summary_ev, summary_p)
    export_best_scheme_to_root(best_scheme)

    # 在 compare.txt 末尾补充“最佳方案”说明
    with open(compare_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 30 + "\n")
        f.write(f"最佳方案：{SCHEME_NAME_CN.get(best_scheme, best_scheme)}\n")
        f.write("判定规则：先比最终总收益；若相同则比最大回撤；再比止损次数；再比最大连续未命中次数。\n")

    print("回测完成。")
    print(f"最佳方案已额外导出到：{config.OUTPUT_DIR}/ 下的 best_* 文件")
    print(f"结果已保存到：{config.OUTPUT_DIR}/EV 和 {config.OUTPUT_DIR}/P")
    print("说明：results.csv 为中文展示版，results_en.csv 为英文原始版。")


if __name__ == "__main__":
    main()
