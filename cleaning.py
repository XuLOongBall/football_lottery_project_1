import pandas as pd
import config


def _log_drop(log, reason, before, after):
    log.append(f"drop {reason}: {before - after} (remain {after})")


def clean_data(df_raw: pd.DataFrame):
    """
    将原始竞彩足球 CSV 清洗为回测可用格式
    """
    log = []
    df = df_raw.copy()

    log.append(f"raw rows: {len(df)}")

    # 1. 比赛时间
    before = len(df)
    df["date"] = pd.to_datetime(df["比赛时间"], errors="coerce")
    df = df.dropna(subset=["date"])
    _log_drop(log, "invalid date", before, len(df))

    # 2. 总进球
    before = len(df)
    df["total_goals"] = pd.to_numeric(df["总进球"], errors="coerce")
    df = df.dropna(subset=["total_goals"])
    _log_drop(log, "invalid total_goals", before, len(df))

    # 转为 int
    df["total_goals"] = df["total_goals"].astype(int)

    # 记录截断次数：>=7 统一视为 7
    cap_cnt = int((df["total_goals"] >= 7).sum())
    df.loc[df["total_goals"] >= 7, "total_goals"] = 7
    log.append(f"cap total_goals>=7 to 7: {cap_cnt}")

    # （可选但建议）过滤负数：如果你不希望出现负数总进球
    before = len(df)
    df = df[df["total_goals"] >= 0]
    _log_drop(log, "total_goals < 0", before, len(df))

    # 3. 赔率（0–7）
    odds_cols = [str(i) for i in range(8)]
    missing = [c for c in odds_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing odds columns: {missing}")

    before = len(df)
    odds_list = []
    valid_mask = []

    for _, row in df.iterrows():
        odds = {}
        valid = True

        for k in range(8):
            try:
                v = float(row[str(k)])
                if v <= 0:
                    valid = False
                    break
                odds[k] = v
            except Exception:
                valid = False
                break

        valid_mask.append(valid)
        odds_list.append(odds)

    df = df[valid_mask].copy()
    _log_drop(log, "invalid odds (non-numeric or <=0)", before, len(df))

    # ✅ odds_list 必须与 df 对齐（更稳）
    odds_list_valid = [od for od, ok in zip(odds_list, valid_mask) if ok]
    df["odds"] = odds_list_valid

    # 4. 主客队（占位，不影响策略）
    df["home_team"] = "HOME"
    df["away_team"] = "AWAY"

    # 5. 月份标识
    df["month_id"] = df["date"].dt.strftime("%Y-%m")

    # 6. 排序
    df = df.sort_values("date").reset_index(drop=True)

    log.append(f"final usable rows: {len(df)}")
    return df, log
