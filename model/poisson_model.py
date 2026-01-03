import numpy as np
from scipy.stats import poisson


def estimate_lambda(history_goals):
    """
    使用历史比赛的总进球数估计泊松分布参数 λ

    Parameters
    ----------
    history_goals : array-like
        历史比赛的总进球数（例如 [2,1,3,0,...]）

    Returns
    -------
    float
        泊松分布的均值参数 λ
    """
    history_goals = np.asarray(history_goals)

    if len(history_goals) == 0:
        raise ValueError("history_goals is empty")

    return history_goals.mean()


def predict_goal_distribution(history_goals, max_goal=7):
    """
    根据历史数据预测下一场比赛的总进球概率分布

    P(G=k) for k=0..6
    P(G>=7) 合并为一类

    Parameters
    ----------
    history_goals : array-like
        当前比赛之前的历史总进球
    max_goal : int
        最大进球数类别（默认 7，表示 >=7）

    Returns
    -------
    probs : dict
        {0: p0, 1: p1, ..., 7: p7}
    """
    lam = estimate_lambda(history_goals)

    probs = {}

    # 0~6 球
    for k in range(max_goal):
        probs[k] = poisson.pmf(k, lam)

    # >=7 球
    probs[max_goal] = 1.0 - sum(probs.values())

    return probs
