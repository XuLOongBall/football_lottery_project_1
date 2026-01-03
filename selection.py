import config

def _p(prob_dist, k: int) -> float:
    return float(prob_dist[k]) if not isinstance(prob_dist, dict) else float(prob_dist[k])

def compute_p_selected(prob_dist, selected_set):
    return sum(_p(prob_dist, k) for k in selected_set)

def compute_ev(prob_dist, odds, selected_set):
    m = len(selected_set)
    return sum(_p(prob_dist, k) * float(odds[k]) for k in selected_set) / m - 1.0

def enumerate_candidate_sets():
    # 仅使用双选候选集合（相邻全套在 config.DOUBLE_CANDIDATES 里）
    return list(config.DOUBLE_CANDIDATES)


def pick_best_set_for_match(match, scheme: str):
    prob_dist = match["prob_dist"]
    odds = match["odds"]

    # 用于正常返回
    best_set = None
    best_p = None
    best_ev = None
    best_score = -1e18

    # 用于 fallback（按p_S最大）
    best_p_set = None
    best_p_p = -1.0
    best_p_ev = None

    for s in enumerate_candidate_sets():
        p_s = compute_p_selected(prob_dist, s)
        ev_s = compute_ev(prob_dist, odds, s)

        # 永远记录“按p_S最好的集合”，供EV方案fallback使用
        if p_s > best_p_p:
            best_p_p = p_s
            best_p_set = s
            best_p_ev = ev_s

        if scheme == "EV":
            # ✅ EV稳健化：命中概率太低的集合不参与EV比较
            if p_s < config.EV_MIN_P:
                continue
            score = ev_s
        else:
            score = p_s

        if score > best_score:
            best_score = score
            best_set = s
            best_p = p_s
            best_ev = ev_s

    # ✅ 若EV过滤后一个集合都没剩：fallback 到按p_S最大
    ev_fallback = 0
    if scheme == "EV" and best_set is None:
        ev_fallback = 1
        best_set = best_p_set
        best_p = best_p_p
        best_ev = best_p_ev
        best_score = best_ev  # 这里写best_ev即可（也可以写best_p，看你想怎么记录）

    rec = match.copy()
    rec["selected_set"] = best_set
    rec["p_selected"] = best_p
    rec["ev_selected"] = best_ev
    rec["score"] = best_score
    rec["scheme"] = scheme
    rec["ev_fallback"] = ev_fallback  # 方便你统计EV方案到底回退了多少次
    return rec


def select_matches_for_month(matches, min_bets, scheme: str):
    scored = [pick_best_set_for_match(m, scheme) for m in matches]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 不足min_bets：返回全部可用（你之前为补次数已经采用过这种策略）
    if len(scored) < min_bets:
        return scored
    return scored[:min_bets]
