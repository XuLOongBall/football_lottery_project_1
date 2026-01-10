import pandas as pd
import config


def _log_drop(log, reason, before, after):
    log.append(f"drop {reason}: {before - after} (remain {after})")


def clean_data(df_raw: pd.DataFrame):
    """
    将原始竞彩足球 CSV 清洗为回测可用格式

    重要改动（为了解决 2020-03 这类“取消/无效场次”导致当月场次不足的问题）：
    - 保留“取消/无效场次”等无赛果比赛作为 void（退票）记录：is_void=1，total_goals 保持 NaN
    - 回测结算时对 void 直接 profit=0（退回本金），hit 记为 NaN（不计入命中率/连错段）
    """
    log = []
    df = df_raw.copy()

    # 清理列名：去掉首尾空格/隐藏 BOM，避免“看起来一样但匹配不到”
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    log.append(f"raw rows: {len(df)}")

    # 1. 时间
    before = len(df)
    df["date"] = pd.to_datetime(df["比赛时间"], errors="coerce")
    df = df.dropna(subset=["date"])
    _log_drop(log, "invalid date", before, len(df))

    # 2. 总进球（允许 void：取消/无效场次/推迟 等）
    #    - played: 必须能得到 total_goals
    #    - void: total_goals 允许 NaN，并打 is_void=1
    df["is_void"] = 0

    before = len(df)
    df["total_goals"] = pd.to_numeric(df.get("总进球"), errors="coerce")

    void_words = ["取消", "无效场次", "推迟", "延期", "腰斩", "中断", "未开赛", "待定"]
    if "全场比分" in df.columns:
        ft = df["全场比分"].astype(str)
        is_void = df["total_goals"].isna() & ft.apply(lambda s: any(w in s for w in void_words))
    else:
        is_void = pd.Series(False, index=df.index)

    df.loc[is_void, "is_void"] = 1

    # 非 void 必须要有 total_goals
    df = df[(df["is_void"] == 1) | (df["total_goals"].notna())].copy()
    _log_drop(log, "invalid total_goals (non-void)", before, len(df))

    # played 的 total_goals 转 int；void 保持 NaN
    played_mask = df["is_void"] == 0
    df.loc[played_mask, "total_goals"] = df.loc[played_mask, "total_goals"].astype(int)

    # 记录截断次数：>=7 统一视为 7（只对 played）
    cap_cnt = int((played_mask & (df["total_goals"] >= 7)).sum())
    df.loc[played_mask & (df["total_goals"] >= 7), "total_goals"] = 7
    log.append(f"cap total_goals>=7 to 7: {cap_cnt}")

    # （可选但建议）过滤负数：只对 played；void 不动
    before = len(df)
    df = df[(df["is_void"] == 1) | (df["total_goals"] >= 0)]
    _log_drop(log, "total_goals < 0 (played only)", before, len(df))

    # 3. 赔率（0–7）
    #    改进：不再强制 0..7 全部存在，只要能形成至少一个可下注的 double set 即可
    odds_cols = [str(i) for i in range(8)]
    missing = [c for c in odds_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing odds columns: {missing}")

    before = len(df)
    odds_list = []
    valid_mask = []

    double_candidates = getattr(config, "DOUBLE_CANDIDATES", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    for _, row in df.iterrows():
        odds = {}
        for k in range(8):
            try:
                v = float(row[str(k)])
                if v > 0:
                    odds[k] = v
            except Exception:
                pass

        feasible = False
        for a, b in double_candidates:
            if a in odds and b in odds:
                feasible = True
                break

        valid_mask.append(feasible)
        odds_list.append(odds)

    df = df[valid_mask].copy()
    _log_drop(log, "invalid odds (no feasible double set)", before, len(df))

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
            parts = s.str.split(r"\s*(?:vs|VS)\s*", n=1, expand=True)
            if parts.shape[1] >= 2:
                df["home_team"] = parts[0].fillna("").str.strip()
                df["away_team"] = parts[1].fillna("").str.strip()
                log.append(f"teams parsed from column: {vs_col}")
                # sample
                try:
                    log.append(f"teams sample: {df['home_team'].head(3).tolist()} vs {df['away_team'].head(3).tolist()}")
                except Exception:
                    pass
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
