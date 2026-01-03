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

    # 回测（注意：run_backtest 需要你改成 run_backtest(df, scheme=...)）
    results = run_backtest(df_clean, scheme=scheme)

    if len(results) == 0:
        raise RuntimeError(f"No betting results produced for scheme={scheme}")

    df_results = pd.DataFrame(results)

    # 保存逐场结果
    df_results.to_csv(
        os.path.join(out_dir, "results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 汇总指标
    summary = compute_metrics(df_results)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # 画收益曲线
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
    with open(os.path.join(config.OUTPUT_DIR, "cleaning_log.txt"), "w") as f:
        for line in clean_log:
            f.write(line + "\n")

    # 3-6. 分别跑两套方案
    summary_ev = run_one_scheme(df_clean, scheme="EV")  # 方案A
    summary_p  = run_one_scheme(df_clean, scheme="P")   # 方案B

    # 7. 输出对比文件（方便写报告）
    compare_path = os.path.join(config.OUTPUT_DIR, "compare.txt")
    with open(compare_path, "w") as f:
        f.write("=== Scheme A: EV ===\n")
        for k, v in summary_ev.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Scheme B: P ===\n")
        for k, v in summary_p.items():
            f.write(f"{k}: {v}\n")

    print("Backtest finished.")
    print(f"Results saved to {config.OUTPUT_DIR}/EV and {config.OUTPUT_DIR}/P")


if __name__ == "__main__":
    main()
