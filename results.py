import numpy as np


def build_result(y_test, pred_test, y_train, pred_train, dates_test, fut, forecast_dates, n_features=0, extras=None):
    y_test = np.asarray(y_test, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)
    err = y_test - pred_test
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / (y_test + 1e-9))) * 100)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_test) + np.abs(pred_test) + 1e-9)) * 100)
    bias = float(np.mean(err))
    ss_r = float(np.sum(err**2))
    ss_t = float(np.sum((y_test - y_test.mean())**2))
    r2 = float(max(0.0, 1 - ss_r / (ss_t + 1e-9)))
    if len(y_test) > 1:
        act_dir = np.sign(np.diff(y_test))
        pred_dir = np.sign(np.diff(pred_test))
        dir_acc = float(np.mean(act_dir == pred_dir) * 100)
    else:
        dir_acc = 0.0

    result = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "bias": bias,
        "r2": r2,
        "dir_acc": dir_acc,
        "yte": y_test.tolist(),
        "pred_test": pred_test.tolist(),
        "pred_train": np.asarray(pred_train, dtype=float).tolist(),
        "train_actual": np.asarray(y_train, dtype=float).tolist(),
        "dates_test": list(dates_test),
        "fut": [float(item) for item in fut],
        "forecast_dates": list(forecast_dates),
        "residuals": err.tolist(),
        "n_features": int(n_features),
        "phase1_history": [],
        "phase2_history": [],
        "selected_features": [],
        "n_selected": 0,
        "lam_curve": [],
        "best_hyper": {},
    }
    if extras:
        result.update(extras)
    return result


def build_final(results, y_full, dates_full):
    best = min(results, key=lambda item: (item["rmse"], 0 if item.get("index") == 12 else 1)) if results else None
    return {
        "dates_full": list(dates_full),
        "Y_full": np.asarray(y_full, dtype=float).tolist(),
        "models": results,
        "best_name": best["name"] if best else None,
        "best_rmse": best["rmse"] if best else None,
    }
