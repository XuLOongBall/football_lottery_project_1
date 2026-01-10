import numpy as np
from scipy.stats import poisson


def estimate_lambda(history_goals):
    """使用历史比赛的总进球数估计泊松分布参数 λ。

    注意：
    - 只使用已结算（非 NaN）的历史总进球
    - history_goals 为空会报错（最早期月份可能需要跳过）
    """
    history_goals = np.asarray(history_goals, dtype=float)
    history_goals = history_goals[~np.isnan(history_goals)]

    if len(history_goals) == 0:
        raise ValueError("history_goals is empty")

    return float(history_goals.mean())


def predict_goal_distribution(history_goals, max_goal=7):
    """根据历史数据预测下一场比赛的总进球概率分布。

    返回 dict: {0: p0, 1: p1, ..., max_goal: p>=max_goal}
    """
    lam = estimate_lambda(history_goals)

    probs = {}
    for k in range(max_goal):
        probs[k] = float(poisson.pmf(k, lam))

    probs[max_goal] = float(1.0 - sum(probs.values()))
    return probs
