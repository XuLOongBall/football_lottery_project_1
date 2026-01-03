from typing import Tuple, Dict
import config


def compute_stake(level: int) -> float:
    """
    根据当前倍投层级计算总下注额
    """
    return config.STAKE_BASE * (config.MARTINGALE_RATIO ** level)


def split_stake_equal(stake: float, selected_set: Tuple[int]) -> Dict[int, float]:
    """
    将总下注额平均分配到每个选项
    """
    per = stake / len(selected_set)
    return {k: per for k in selected_set}


def settle_bet(
    selected_set: Tuple[int],
    odds: Dict[int, float],
    y_true: int,
    level: int
):
    """
    结算单场投注 + 严格止损：
    - 连续不中最多允许 config.MAX_MARTINGALE_LEVEL 次（例如 8）
    - 触发止损后 next_level 重置为 0
    """
    max_losses = config.MAX_MARTINGALE_LEVEL          # 例如 8（最大允许连续不中次数）
    max_level = max_losses - 1                        # 对应可下注的最大层级：0..7

    # 安全保护：不允许 level 超过 max_level
    level = min(level, max_level)

    stake_total = compute_stake(level)
    stake_split = split_stake_equal(stake_total, selected_set)

    hit = int(y_true in selected_set)
    stoploss = 0
    profit = 0.0

    if hit:
        for k, s in stake_split.items():
            if k == y_true:
                profit += s * (odds[k] - 1)
            else:
                profit -= s
        next_level = 0
    else:
        profit = -stake_total

        # 关键：如果当前已经在 max_level 且还不中，
        # 说明这是本轮第 max_losses 次不中 -> 触发止损，重置
        if level >= max_level:
            stoploss = 1
            next_level = 0
        else:
            next_level = level + 1
    return {
        "stake_total": stake_total,
        "stake_split": stake_split,
        "hit": hit,
        "profit": profit,
        "next_level": next_level,
        "stoploss": stoploss,
    }