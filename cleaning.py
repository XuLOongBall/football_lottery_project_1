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

    # 清理列名：去掉首尾空格/隐藏 BOM，避免“看起来一样但匹配不到”
    df.columns = df.columns.astype(str).map(lambda x: x.replace("\ufeff", "").strip())

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

    # 4. 主客队（优先从原始列提取；否则从“主客场/对阵”列解析；最后才占位）
    home_candidates = ["主队", "主队名称", "主队名", "主队简称", "HomeTeam", "home_team"]
    away_candidates = ["客队", "客队名称", "客队名", "客队简称", "AwayTeam", "away_team"]

    def _first_existing(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    home_col = _first_existing(home_candidates)
    away_col = _first_existing(away_candidates)

    if home_col and away_col:
        df["home_team"] = df[home_col].astype(str).str.strip()
        df["away_team"] = df[away_col].astype(str).str.strip()
        log.append(f"teams from columns: home={home_col}, away={away_col}")
    else:
        # 你这份数据里“主客场”列形如 “A VS B” ：A=主队，B=客队
        vs_candidates = ["主客场", "对阵", "比赛", "赛事", "match", "Match", "VS", "vs"]
        vs_col = _first_existing(vs_candidates)

        if vs_col:
            s = df[vs_col].astype(str)
            # 兼容各种分隔符：VS / vs / - / — / ：等
            parts = s.str.split(r"\s*(?:vs|VS|－|-|—|–|:|：)\s*", n=1, expand=True)
            if parts.shape[1] >= 2:
                df["home_team"] = parts[0].fillna("").str.strip()
                df["away_team"] = parts[1].fillna("").str.strip()
                log.append(f"teams parsed from column: {vs_col}")
                log.append(f"teams sample: {df['home_team'].head(3).tolist()} vs {df['away_team'].head(3).tolist()}")
            else:
                df["home_team"] = "HOME"
                df["away_team"] = "AWAY"
                log.append(f"teams placeholder: failed to parse '{vs_col}'")
        else:
            df["home_team"] = "HOME"
            df["away_team"] = "AWAY"
            log.append("teams placeholder: no team columns found")

    # 5. 月份标识
    df["month_id"] = df["date"].dt.strftime("%Y-%m")

    # 6. 排序
    df = df.sort_values("date").reset_index(drop=True)

    log.append(f"final usable rows: {len(df)}")
    return df, log
