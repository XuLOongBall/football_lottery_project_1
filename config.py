# =========================
# Data configuration
# =========================
DATA_FILE = "data/sport_lottery_datas.csv"
DATA_START = "2015-01-01"
DATA_END = "2024-12-31"

# =========================
# Model configuration
# =========================
MODEL_NAME = "poisson_v1"
ROLLING_WINDOW = 200   # 用过去多少场比赛估计 λ

# =========================
# Selection constraints
# =========================
MONTHLY_MIN_BETS = 16
TOTAL_MIN_BETS = 1800

# =========================
# Betting configuration
# =========================
STAKE_BASE = 100
MARTINGALE_RATIO = 3
MAX_MARTINGALE_LEVEL = 8

EV_MIN_P = 0.30   # EV方案候选集合的命中概率下限（建议先从0.30开始）
EV_FALLBACK = "P" # 若所有集合都达不到EV_MIN_P，则回退到按p_S最大来选


BET_MODE = "double"  # "single" or "double"
DOUBLE_CANDIDATES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

ALLOCATION_MODE = "equal"  # "equal" or "prob"

# =========================
# Runtime / output
# =========================
RANDOM_SEED = 42
OUTPUT_DIR = "outputs"
