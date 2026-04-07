import traceback

import numpy as np

from .constants import MODEL_NAMES
from .models import (
    decode_elm_params,
    decode_gbdt_params,
    decode_lstm_params,
    decode_novelty_params,
    eval_elm_candidate,
    eval_gbdt_candidate,
    eval_lstm_candidate,
    run_elm,
    run_gbdt,
    run_lstm,
)
from .optimizers import adaptive_binary_hybrid, gwo_optimize, hybrid_pso_gwo_optimize, pso_optimize, trim_cfg
from .results import build_final, build_result
from .state import jobs


def log_msg(job_id, msg):
    jobs[job_id]["log"].append(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def run_pipeline(job_id, df, cfg):
    try:
        base_cfg = {
            "pop_size": int(cfg.get("popSize", 20)),
            "max_iter": int(cfg.get("maxIter", 40)),
            "c1": float(cfg.get("c1", 1.5)),
            "c2": float(cfg.get("c2", 2.0)),
            "w": float(cfg.get("w", 0.72)),
        }
        dataset = _build_dataset(df, cfg)
        tune_seq_cfg = trim_cfg(base_cfg, pop_scale=0.35, iter_scale=0.3, min_pop=4, min_iter=2, max_eval=10)
        tune_flat_cfg = trim_cfg(base_cfg, pop_scale=0.35, iter_scale=0.3, min_pop=4, min_iter=2, max_eval=8)
        novelty_phase1_cfg = trim_cfg(base_cfg, pop_scale=0.45, iter_scale=0.35, min_pop=5, min_iter=3, max_eval=15)
        novelty_phase2_cfg = trim_cfg(base_cfg, pop_scale=0.45, iter_scale=0.35, min_pop=5, min_iter=3, max_eval=18)
        results = []
        tuned_cache = {}

        def get_tuned(label, builder):
            if label not in tuned_cache:
                log_msg(job_id, f"Tuning {label}")
                tuned_cache[label] = builder()
            return tuned_cache[label]

        model_runs = [
            lambda: run_lstm(dataset, {"seq_len": dataset["base_seq_len"]}),
            lambda: run_gbdt(dataset, {}),
            lambda: run_elm(dataset, {}),
            lambda: run_lstm(dataset, get_tuned("LSTM + PSO", lambda: _tune_lstm(dataset, tune_seq_cfg, "pso"))),
            lambda: run_lstm(dataset, get_tuned("LSTM + GWO", lambda: _tune_lstm(dataset, tune_seq_cfg, "gwo"))),
            lambda: run_elm(dataset, get_tuned("ELM + PSO", lambda: _tune_elm(dataset, tune_flat_cfg, "pso"))),
            lambda: run_elm(dataset, get_tuned("ELM + GWO", lambda: _tune_elm(dataset, tune_flat_cfg, "gwo"))),
            lambda: run_gbdt(dataset, get_tuned("GBDT + PSO", lambda: _tune_gbdt(dataset, tune_flat_cfg, "pso"))),
            lambda: run_gbdt(dataset, get_tuned("GBDT + GWO", lambda: _tune_gbdt(dataset, tune_flat_cfg, "gwo"))),
            lambda: run_lstm(dataset, get_tuned("LSTM + PSO-GWO", lambda: _tune_lstm(dataset, tune_seq_cfg, "hybrid"))),
            lambda: run_elm(dataset, get_tuned("ELM + PSO-GWO", lambda: _tune_elm(dataset, tune_flat_cfg, "hybrid"))),
            lambda: run_gbdt(dataset, get_tuned("GBDT + PSO-GWO", lambda: _tune_gbdt(dataset, tune_flat_cfg, "hybrid"))),
            lambda: _run_adaptive_dual_phase(
                dataset,
                novelty_phase1_cfg,
                novelty_phase2_cfg,
                job_id,
                get_tuned("LSTM + PSO-GWO", lambda: _tune_lstm(dataset, tune_seq_cfg, "hybrid")),
            ),
        ]

        for idx, name in enumerate(MODEL_NAMES):
            log_msg(job_id, f"Training [{idx + 1}/{len(MODEL_NAMES)}] {name}")
            jobs[job_id]["progress"] = idx / len(MODEL_NAMES)
            try:
                result = model_runs[idx]()
                result["name"] = name
                result["index"] = idx
                results.append(result)
                jobs[job_id]["results"].append(result)
                jobs[job_id]["progress"] = (idx + 1) / len(MODEL_NAMES)
                log_msg(job_id, f"Done {name} RMSE={result['rmse']:.4f} R2={result['r2']:.4f}")
            except Exception as err:
                log_msg(job_id, f"Fail {name}: {err}")
                traceback.print_exc()

        jobs[job_id]["final"] = build_final(results, dataset["y_full"], dataset["dates_full"])
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 1.0
        log_msg(job_id, f"Complete {len(results)}/{len(MODEL_NAMES)} models")
    except Exception as err:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log"].append(f"FATAL: {err}")
        traceback.print_exc()


def _build_dataset(df, cfg):
    seq_len = int(cfg.get("seqLen", 40))
    test_pct = float(cfg.get("testPct", 0.2))
    feature_cols = [col for col in df.columns if col not in ("Date", "Close")]
    x_raw = df[feature_cols].to_numpy(dtype=float)
    y_full = df["Close"].to_numpy(dtype=float)
    dates_full = df["Date"].tolist()
    size = len(y_full)
    n_test = max(16, int(round(size * test_pct)))
    n_train = size - n_test
    if n_train <= seq_len + 40 or n_test <= 0:
        raise ValueError("Sequence length is too large for the available data.")

    feature_mean = x_raw[:n_train].mean(axis=0)
    feature_std = x_raw[:n_train].std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    x_norm = (x_raw - feature_mean) / feature_std

    target_mean = float(y_full[:n_train].mean())
    target_scale = float(max(y_full[:n_train].std(), 1e-6))
    y_scaled = (y_full - target_mean) / target_scale

    tune_rows = min(n_train, max(140, seq_len * 4))
    tune_start = n_train - tune_rows
    return {
        "feature_cols": feature_cols,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "x_train_flat": x_norm[:n_train],
        "x_test_flat": x_norm[n_train:],
        "x_tune_flat": x_norm[tune_start:n_train],
        "y_train_scaled": y_scaled[:n_train],
        "y_test_scaled": y_scaled[n_train:],
        "y_tune_scaled": y_scaled[tune_start:n_train],
        "y_train": y_full[:n_train],
        "y_test": y_full[n_train:],
        "y_full": y_full,
        "dates_test": dates_full[n_train:],
        "dates_full": dates_full,
        "history_raw": df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True),
        "base_seq_len": seq_len,
        "n_future": 30,
    }


def _tune_lstm(dataset, cfg, method):
    bounds = [(48, 176), (0.2, 0.7), (1e-4, 4e-3), (1e-6, 8e-3), (24, 96)]

    def objective(values):
        params = decode_lstm_params(values, dataset["base_seq_len"])
        params["tune_iter"] = 25
        return eval_lstm_candidate(dataset, params)

    result = _run_optimizer(bounds, objective, cfg, method, seed=11)
    params = decode_lstm_params(result["vector"], dataset["base_seq_len"])
    params["fit_iter"] = 100
    return params


def _tune_gbdt(dataset, cfg, method):
    bounds = [(80, 360), (0.01, 0.25), (2, 8), (8, 60), (0.0, 0.3)]
    result = _run_optimizer(bounds, lambda values: eval_gbdt_candidate(dataset, decode_gbdt_params(values)), cfg, method, seed=17)
    return decode_gbdt_params(result["vector"])


def _tune_elm(dataset, cfg, method):
    bounds = [(40, 360), (1e-5, 1.2), (0.05, 0.8)]
    result = _run_optimizer(bounds, lambda values: eval_elm_candidate(dataset, decode_elm_params(values)), cfg, method, seed=23)
    return decode_elm_params(result["vector"])


def _run_adaptive_dual_phase(dataset, phase1_cfg, phase2_cfg, job_id, companion_params):
    feature_cols = dataset["feature_cols"]
    core_names = {"Open", "High", "Low", "Volume", "LogReturn", "CloseLag1", "RetLag1", "SMA10", "EMA20", "RSI14", "MACD", "ATR14"}
    core_idx = [idx for idx, name in enumerate(feature_cols) if name in core_names]
    log_msg(job_id, "Novelty phase 1 feature search")

    def phase1_objective(mask):
        selected = np.where(mask > 0.5)[0]
        selected = np.unique(np.concatenate([selected, core_idx]))
        if len(selected) < max(8, len(core_idx)):
            return 1e9
        probe = {
            "n_hidden": 72,
            "n_hidden2": 28,
            "lr": 0.0012,
            "alpha": 0.0004,
            "batch_size": 32,
            "seq_len": dataset["base_seq_len"],
            "tune_iter": 30,
        }
        penalty = 0.018 * max(0, len(selected) - len(core_idx))
        return eval_lstm_candidate(dataset, probe, selected) + penalty

    phase1 = adaptive_binary_hybrid(len(feature_cols), phase1_objective, phase1_cfg, seed=29, locked_idx=core_idx)
    selected_idx = np.where(phase1["binary"] > 0.5)[0]
    selected_idx = np.unique(np.concatenate([selected_idx, core_idx]))
    selected_names = [feature_cols[idx] for idx in selected_idx]
    log_msg(job_id, f"Novelty phase 1 picked {len(selected_idx)} features")

    low_seq = max(18, int(round(dataset["base_seq_len"] * 0.6)))
    high_seq = min(90, int(round(dataset["base_seq_len"] * 1.5)))
    bounds = [(56, 176), (0.25, 0.75), (1e-4, 4e-3), (1e-6, 8e-3), (24, 96), (low_seq, high_seq)]
    log_msg(job_id, "Novelty phase 2 hyper search")
    phase2 = hybrid_pso_gwo_optimize(
        bounds,
        lambda values: eval_lstm_candidate(dataset, decode_novelty_params(values, dataset["base_seq_len"]), selected_idx),
        phase2_cfg,
        seed=31,
        adaptive=True,
    )
    params = decode_novelty_params(phase2["vector"], dataset["base_seq_len"])
    params["feature_idx"] = selected_idx
    selected_result = run_lstm(dataset, params)
    companion_result = run_lstm(dataset, companion_params)
    selected_score = max(eval_lstm_candidate(dataset, params, selected_idx), 1e-6)
    companion_score = max(eval_lstm_candidate(dataset, companion_params), 1e-6)
    final_result = companion_result if companion_score <= selected_score else _blend_results(selected_result, companion_result, 1 / selected_score, 1 / companion_score)
    final_result["phase1_history"] = phase1["history"]
    final_result["phase2_history"] = phase2["history"]
    final_result["selected_features"] = selected_names
    final_result["n_selected"] = len(selected_idx)
    final_result["lam_curve"] = phase2["lam_curve"]
    final_result["best_hyper"] = {
        "features": len(selected_idx),
        "seq_len": params["seq_len"],
        "n_hidden": params["n_hidden"],
        "n_hidden2": params["n_hidden2"],
        "lr": params["lr"],
        "alpha": params["alpha"],
        "batch_size": params["batch_size"],
        "companion_seq_len": companion_params["seq_len"],
        "companion_bias": round(1.0 if companion_score <= selected_score else (1 / companion_score) / ((1 / selected_score) + (1 / companion_score)), 3),
    }
    return final_result


def _run_optimizer(bounds, objective, cfg, method, seed):
    if method == "pso":
        return pso_optimize(bounds, objective, cfg, seed=seed)
    if method == "gwo":
        return gwo_optimize(bounds, objective, cfg, seed=seed)
    return hybrid_pso_gwo_optimize(bounds, objective, cfg, seed=seed, adaptive=False)


def _blend_results(primary, companion, w_primary, w_companion):
    weight_sum = w_primary + w_companion
    p_weight = w_primary / weight_sum
    c_weight = w_companion / weight_sum
    train_len = min(len(primary["pred_train"]), len(companion["pred_train"]), len(primary["train_actual"]), len(companion["train_actual"]))
    pred_train = p_weight * np.array(primary["pred_train"][-train_len:]) + c_weight * np.array(companion["pred_train"][-train_len:])
    train_actual = primary["train_actual"][-train_len:]
    pred_test = p_weight * np.array(primary["pred_test"]) + c_weight * np.array(companion["pred_test"])
    fut = p_weight * np.array(primary["fut"]) + c_weight * np.array(companion["fut"])
    return build_result(
        primary["yte"],
        pred_test,
        train_actual,
        pred_train,
        primary["dates_test"],
        fut,
        primary["forecast_dates"],
        n_features=primary["n_features"],
    )
