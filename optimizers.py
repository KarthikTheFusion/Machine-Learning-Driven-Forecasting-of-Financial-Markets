import numpy as np


def pso_optimize(bounds, objective, cfg, seed=42):
    lb, ub = _bounds(bounds)
    rng = np.random.default_rng(seed)
    pop = _spawn(rng, lb, ub, cfg["pop_size"])
    vel = np.zeros_like(pop)
    p_best = pop.copy()
    p_fit = np.array([objective(item) for item in pop], dtype=float)
    best_idx = int(np.argmin(p_fit))
    g_best = p_best[best_idx].copy()
    g_fit = float(p_fit[best_idx])
    history = [g_fit]

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        inertia = float(cfg["w"] * (1 - ratio) + 0.38 * ratio)
        r1 = rng.random(pop.shape)
        r2 = rng.random(pop.shape)
        vel = inertia * vel + cfg["c1"] * r1 * (p_best - pop) + cfg["c2"] * r2 * (g_best - pop)
        pop = np.clip(pop + vel, lb, ub)
        fit = np.array([objective(item) for item in pop], dtype=float)
        improved = fit < p_fit
        if np.any(improved):
            p_best[improved] = pop[improved]
            p_fit[improved] = fit[improved]
            best_idx = int(np.argmin(p_fit))
            if p_fit[best_idx] < g_fit:
                g_fit = float(p_fit[best_idx])
                g_best = p_best[best_idx].copy()
        history.append(g_fit)

    return {"vector": g_best, "fitness": g_fit, "history": history}


def gwo_optimize(bounds, objective, cfg, seed=42):
    lb, ub = _bounds(bounds)
    rng = np.random.default_rng(seed)
    wolves = _spawn(rng, lb, ub, cfg["pop_size"])
    fit = np.array([objective(item) for item in wolves], dtype=float)
    alpha, beta, delta, alpha_fit = _leaders(wolves, fit)
    history = [alpha_fit]

    for step in range(cfg["max_iter"]):
        a = 2 - 2 * step / max(1, cfg["max_iter"] - 1)
        for idx in range(len(wolves)):
            x1 = _encircle(rng, wolves[idx], alpha, a)
            x2 = _encircle(rng, wolves[idx], beta, a)
            x3 = _encircle(rng, wolves[idx], delta, a)
            wolves[idx] = np.clip((x1 + x2 + x3) / 3, lb, ub)

        fit = np.array([objective(item) for item in wolves], dtype=float)
        alpha, beta, delta, alpha_fit = _leaders(wolves, fit)
        history.append(alpha_fit)

    return {"vector": alpha, "fitness": alpha_fit, "history": history}


def hybrid_pso_gwo_optimize(bounds, objective, cfg, seed=42, adaptive=False):
    lb, ub = _bounds(bounds)
    rng = np.random.default_rng(seed)
    pop = _spawn(rng, lb, ub, cfg["pop_size"])
    vel = np.zeros_like(pop)
    p_best = pop.copy()
    p_fit = np.array([objective(item) for item in pop], dtype=float)
    best_idx = int(np.argmin(p_fit))
    g_best = p_best[best_idx].copy()
    g_fit = float(p_fit[best_idx])
    history = [g_fit]
    lam_curve = []

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        lam = float(1 - ratio**2) if adaptive else 0.5
        a = 2 - 2 * ratio
        order = np.argsort(p_fit)
        alpha = p_best[order[0]]
        beta = p_best[order[1]]
        delta = p_best[order[2]]
        lam_curve.append(lam)

        for idx in range(len(pop)):
            r1 = rng.random(pop.shape[1])
            r2 = rng.random(pop.shape[1])
            vel[idx] = cfg["w"] * vel[idx] + cfg["c1"] * r1 * (p_best[idx] - pop[idx]) + cfg["c2"] * r2 * (g_best - pop[idx])
            pso_pos = pop[idx] + vel[idx]
            gwo_pos = (_encircle(rng, pop[idx], alpha, a) + _encircle(rng, pop[idx], beta, a) + _encircle(rng, pop[idx], delta, a)) / 3
            pop[idx] = np.clip(lam * pso_pos + (1 - lam) * gwo_pos, lb, ub)
            fit = float(objective(pop[idx]))
            if fit < p_fit[idx]:
                p_best[idx] = pop[idx].copy()
                p_fit[idx] = fit
                if fit < g_fit:
                    g_fit = fit
                    g_best = pop[idx].copy()

        history.append(g_fit)

    return {"vector": g_best, "fitness": g_fit, "history": history, "lam_curve": lam_curve}


def adaptive_binary_hybrid(dim, objective, cfg, seed=42, locked_idx=None):
    rng = np.random.default_rng(seed)
    pop = rng.random((cfg["pop_size"], dim))
    vel = np.zeros_like(pop)
    locked_idx = np.array(sorted(locked_idx or []), dtype=int)

    def to_mask(values):
        transfer = np.abs(np.tanh(values))
        mask = (rng.random(dim) < transfer).astype(float)
        if locked_idx.size:
            mask[locked_idx] = 1.0
        return mask

    masks = np.array([to_mask(item) for item in pop])
    p_best = pop.copy()
    p_mask = masks.copy()
    p_fit = np.array([objective(item) for item in masks], dtype=float)
    best_idx = int(np.argmin(p_fit))
    g_best = p_best[best_idx].copy()
    g_mask = p_mask[best_idx].copy()
    g_fit = float(p_fit[best_idx])
    history = [g_fit]
    lam_curve = []

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        lam = float(1 - ratio**2)
        a = 2 - 2 * ratio
        order = np.argsort(p_fit)
        alpha = p_best[order[0]]
        beta = p_best[order[1]]
        delta = p_best[order[2]]
        lam_curve.append(lam)

        for idx in range(len(pop)):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            vel[idx] = cfg["w"] * vel[idx] + cfg["c1"] * r1 * (p_best[idx] - pop[idx]) + cfg["c2"] * r2 * (g_best - pop[idx])
            pso_pos = pop[idx] + vel[idx]
            gwo_pos = (_encircle(rng, pop[idx], alpha, a) + _encircle(rng, pop[idx], beta, a) + _encircle(rng, pop[idx], delta, a)) / 3
            pop[idx] = np.clip(lam * pso_pos + (1 - lam) * gwo_pos, 0.0, 1.0)
            mask = to_mask(pop[idx])
            fit = float(objective(mask))
            if fit < p_fit[idx]:
                p_best[idx] = pop[idx].copy()
                p_mask[idx] = mask
                p_fit[idx] = fit
                if fit < g_fit:
                    g_fit = fit
                    g_best = pop[idx].copy()
                    g_mask = mask.copy()

        history.append(g_fit)

    return {"vector": g_best, "binary": g_mask, "fitness": g_fit, "history": history, "lam_curve": lam_curve}


def trim_cfg(cfg, pop_scale=1.0, iter_scale=1.0, min_pop=6, min_iter=4, max_eval=64):
    pop_size = max(min_pop, int(round(cfg["pop_size"] * pop_scale)))
    max_iter = max(min_iter, int(round(cfg["max_iter"] * iter_scale)))
    if pop_size * max_iter > max_eval:
        max_iter = max(min_iter, max_eval // pop_size)
    return {
        "pop_size": max(4, pop_size),
        "max_iter": max(2, max_iter),
        "c1": float(cfg["c1"]),
        "c2": float(cfg["c2"]),
        "w": float(cfg["w"]),
    }


def _bounds(bounds):
    lb = np.array([item[0] for item in bounds], dtype=float)
    ub = np.array([item[1] for item in bounds], dtype=float)
    return lb, ub


def _spawn(rng, lb, ub, size):
    return lb + rng.random((size, len(lb))) * (ub - lb)


def _encircle(rng, current, leader, a):
    leader_a = 2 * a * rng.random(len(current)) - a
    leader_c = 2 * rng.random(len(current))
    return leader - leader_a * np.abs(leader_c * leader - current)


def _leaders(pop, fit):
    order = np.argsort(fit)
    alpha = pop[order[0]].copy()
    beta = pop[order[1]].copy()
    delta = pop[order[2]].copy()
    return alpha, beta, delta, float(fit[order[0]])
