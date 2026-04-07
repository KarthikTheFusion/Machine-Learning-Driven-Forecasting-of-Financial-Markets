import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from .features import extend_history, next_business_date, prepare_features
from .results import build_result


def decode_lstm_params(values, seq_len):
    n_hidden = int(np.clip(round(values[0]), 48, 176))
    ratio = float(np.clip(values[1], 0.2, 0.7))
    n_hidden2 = int(max(10, round(n_hidden * ratio)))
    lr = float(np.clip(values[2], 1e-4, 4e-3))
    alpha = float(np.clip(values[3], 1e-6, 8e-3))
    batch_size = int(np.clip(round(values[4] / 8) * 8, 24, 96))
    return {
        "n_hidden": n_hidden,
        "n_hidden2": n_hidden2,
        "lr": lr,
        "alpha": alpha,
        "batch_size": batch_size,
        "seq_len": int(seq_len),
    }


def decode_novelty_params(values, base_seq_len):
    low_seq = max(18, int(round(base_seq_len * 0.6)))
    high_seq = min(90, int(round(base_seq_len * 1.5)))
    params = decode_lstm_params(values[:5], base_seq_len)
    params["seq_len"] = int(np.clip(round(values[5]), low_seq, high_seq))
    params["ensemble"] = 2
    params["fit_iter"] = 160
    params["tune_iter"] = 35
    return params


def decode_gbdt_params(values):
    return {
        "n_estimators": int(np.clip(round(values[0]), 80, 360)),
        "lr": float(np.clip(values[1], 0.01, 0.25)),
        "max_depth": int(np.clip(round(values[2]), 2, 8)),
        "min_samples_leaf": int(np.clip(round(values[3]), 8, 60)),
        "l2": float(np.clip(values[4], 0.0, 0.3)),
    }


def decode_elm_params(values):
    return {
        "n_hidden": int(np.clip(round(values[0]), 40, 360)),
        "alpha": float(np.clip(values[1], 1e-5, 2.0)),
        "scale": float(np.clip(values[2], 0.05, 1.0)),
    }


def eval_lstm_candidate(dataset, params, feature_idx=None):
    feature_idx = _feature_idx(dataset, feature_idx)
    x_values = dataset["x_tune_flat"][:, feature_idx]
    y_values = dataset["y_tune_scaled"]
    x_seq, y_seq = build_sequences(x_values, y_values, params["seq_len"])
    split = _split_point(len(x_seq), min_train=max(42, params["seq_len"]), min_val=16)
    if split is None:
        return 1e9
    model = _fit_sequence_model(x_seq[:split], y_seq[:split], params, seed=42, max_iter=params.get("tune_iter", 110))
    pred = model.predict(x_seq[split:].reshape(len(x_seq) - split, -1))
    return _rmse(y_seq[split:], pred) * dataset["target_scale"]


def eval_gbdt_candidate(dataset, params):
    split = _split_point(len(dataset["x_tune_flat"]), min_train=60, min_val=18)
    if split is None:
        return 1e9
    model = _fit_gbdt(dataset["x_tune_flat"][:split], dataset["y_tune_scaled"][:split], params)
    pred = model.predict(dataset["x_tune_flat"][split:])
    return _rmse(dataset["y_tune_scaled"][split:], pred) * dataset["target_scale"]


def eval_elm_candidate(dataset, params):
    split = _split_point(len(dataset["x_tune_flat"]), min_train=60, min_val=18)
    if split is None:
        return 1e9
    model = _fit_elm(dataset["x_tune_flat"][:split], dataset["y_tune_scaled"][:split], params, seed=42)
    pred = _predict_elm(model, dataset["x_tune_flat"][split:])
    return _rmse(dataset["y_tune_scaled"][split:], pred) * dataset["target_scale"]


def run_lstm(dataset, hyper):
    params = {
        "n_hidden": int(hyper.get("n_hidden", 72)),
        "n_hidden2": int(hyper.get("n_hidden2", 28)),
        "lr": float(hyper.get("lr", 9e-4)),
        "alpha": float(hyper.get("alpha", 3e-4)),
        "batch_size": int(hyper.get("batch_size", 32)),
        "seq_len": int(hyper.get("seq_len", dataset["base_seq_len"])),
        "ensemble": int(hyper.get("ensemble", 1)),
        "fit_iter": int(hyper.get("fit_iter", 100)),
    }
    feature_idx = _feature_idx(dataset, hyper.get("feature_idx"))
    x_train = dataset["x_train_flat"][:, feature_idx]
    x_test = dataset["x_test_flat"][:, feature_idx]
    x_train_seq, y_train_seq = build_sequences(x_train, dataset["y_train_scaled"], params["seq_len"])
    x_test_seq, y_test_seq = build_test_sequences(x_train, x_test, dataset["y_train_scaled"], dataset["y_test_scaled"], params["seq_len"])
    if len(x_train_seq) == 0 or len(x_test_seq) == 0:
        raise ValueError("Sequence length is too large for the available data.")

    models = [
        _fit_sequence_model(x_train_seq, y_train_seq, params, seed=42 + idx, max_iter=params["fit_iter"])
        for idx in range(params["ensemble"])
    ]
    pred_train = _inverse(dataset, _predict_sequence(models, x_train_seq))
    pred_test = _inverse(dataset, _predict_sequence(models, x_test_seq))
    fut, forecast_dates = _forecast_sequence(models, dataset, feature_idx, params["seq_len"])
    extras = {"best_hyper": {key: value for key, value in params.items() if key != "ensemble"}}
    return build_result(
        dataset["y_test"],
        pred_test,
        dataset["y_train"][params["seq_len"]:],
        pred_train,
        dataset["dates_test"],
        fut,
        forecast_dates,
        n_features=len(feature_idx),
        extras=extras,
    )


def run_gbdt(dataset, hyper):
    params = {
        "n_estimators": int(hyper.get("n_estimators", 140)),
        "lr": float(hyper.get("lr", 0.05)),
        "max_depth": int(hyper.get("max_depth", 3)),
        "min_samples_leaf": int(hyper.get("min_samples_leaf", 20)),
        "l2": float(hyper.get("l2", 0.03)),
    }
    model = _fit_gbdt(dataset["x_train_flat"], dataset["y_train_scaled"], params)
    pred_train = _inverse(dataset, model.predict(dataset["x_train_flat"]))
    pred_test = _inverse(dataset, model.predict(dataset["x_test_flat"]))
    fut, forecast_dates = _forecast_flat(model, dataset)
    return build_result(
        dataset["y_test"],
        pred_test,
        dataset["y_train"],
        pred_train,
        dataset["dates_test"],
        fut,
        forecast_dates,
        n_features=len(dataset["feature_cols"]),
        extras={"best_hyper": params},
    )


def run_elm(dataset, hyper):
    params = {
        "n_hidden": int(hyper.get("n_hidden", 160)),
        "alpha": float(hyper.get("alpha", 0.05)),
        "scale": float(hyper.get("scale", 0.22)),
    }
    model = _fit_elm(dataset["x_train_flat"], dataset["y_train_scaled"], params, seed=42)
    pred_train = _inverse(dataset, _predict_elm(model, dataset["x_train_flat"]))
    pred_test = _inverse(dataset, _predict_elm(model, dataset["x_test_flat"]))
    fut, forecast_dates = _forecast_flat(model, dataset, elm=True)
    return build_result(
        dataset["y_test"],
        pred_test,
        dataset["y_train"],
        pred_train,
        dataset["dates_test"],
        fut,
        forecast_dates,
        n_features=len(dataset["feature_cols"]),
        extras={"best_hyper": params},
    )


def build_sequences(x_frame, y_values, seq_len):
    if len(x_frame) <= seq_len:
        return np.empty((0, seq_len, x_frame.shape[1])), np.empty(0)
    x_seq = np.stack([x_frame[idx - seq_len:idx] for idx in range(seq_len, len(x_frame))])
    y_seq = y_values[seq_len:]
    return x_seq, y_seq


def build_test_sequences(x_train, x_test, y_train, y_test, seq_len):
    joined_x = np.vstack([x_train[-seq_len:], x_test])
    joined_y = np.concatenate([y_train[-seq_len:], y_test])
    x_seq, y_seq = build_sequences(joined_x, joined_y, seq_len)
    return x_seq, y_seq


def _feature_idx(dataset, feature_idx):
    if feature_idx is None:
        return np.arange(len(dataset["feature_cols"]))
    return np.array(feature_idx, dtype=int)


def _fit_sequence_model(x_train_seq, y_train, params, seed, max_iter):
    batch_size = int(min(max(16, params["batch_size"]), len(x_train_seq)))
    hidden = (params["n_hidden"], max(8, params["n_hidden2"]))
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=params["alpha"],
        batch_size=batch_size,
        learning_rate="adaptive",
        learning_rate_init=params["lr"],
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.14,
        n_iter_no_change=18,
        random_state=seed,
        verbose=False,
    )
    model.fit(x_train_seq.reshape(len(x_train_seq), -1), y_train)
    return model


def _predict_sequence(models, x_seq):
    flat = x_seq.reshape(len(x_seq), -1)
    preds = np.column_stack([model.predict(flat) for model in models])
    return preds.mean(axis=1)


def _fit_gbdt(x_train, y_train, params):
    model = HistGradientBoostingRegressor(
        learning_rate=params["lr"],
        max_depth=params["max_depth"],
        max_iter=params["n_estimators"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2"],
        early_stopping=False,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _fit_elm(x_train, y_train, params, seed):
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, params["scale"], size=(x_train.shape[1], params["n_hidden"]))
    bias = rng.normal(0.0, params["scale"], size=(params["n_hidden"],))
    hidden = np.tanh(x_train @ weights + bias)
    ridge = params["alpha"] * np.eye(params["n_hidden"])
    beta = np.linalg.pinv(hidden.T @ hidden + ridge) @ hidden.T @ y_train
    return {"weights": weights, "bias": bias, "beta": beta}


def _predict_elm(model, x_frame):
    hidden = np.tanh(x_frame @ model["weights"] + model["bias"])
    return hidden @ model["beta"]


def _forecast_sequence(models, dataset, feature_idx, seq_len):
    raw = dataset["history_raw"].copy()
    fut = []
    dates = []
    for _ in range(dataset["n_future"]):
        feat = prepare_features(raw)
        x_values = feat[dataset["feature_cols"]].to_numpy(dtype=float)
        x_norm = (x_values - dataset["feature_mean"]) / dataset["feature_std"]
        x_seq = x_norm[:, feature_idx]
        if len(x_seq) <= seq_len:
            break
        pred = float(_inverse(dataset, _predict_sequence(models, x_seq[-seq_len:].reshape(1, seq_len, len(feature_idx))))[0])
        dates.append(next_business_date(raw["Date"].iloc[-1]))
        fut.append(pred)
        raw = extend_history(raw, pred)
    return fut, dates


def _forecast_flat(model, dataset, elm=False):
    raw = dataset["history_raw"].copy()
    fut = []
    dates = []
    for _ in range(dataset["n_future"]):
        feat = prepare_features(raw)
        x_values = feat[dataset["feature_cols"]].to_numpy(dtype=float)
        x_norm = (x_values - dataset["feature_mean"]) / dataset["feature_std"]
        row = x_norm[-1:]
        pred_scaled = _predict_elm(model, row)[0] if elm else model.predict(row)[0]
        pred = float(_inverse(dataset, np.array([pred_scaled]))[0])
        dates.append(next_business_date(raw["Date"].iloc[-1]))
        fut.append(pred)
        raw = extend_history(raw, pred)
    return fut, dates


def _split_point(length, min_train, min_val):
    if length < min_train + min_val:
        return None
    split = int(round(length * 0.8))
    split = max(split, min_train)
    if length - split < min_val:
        split = length - min_val
    return split if split >= min_train else None


def _inverse(dataset, values):
    return np.asarray(values, dtype=float) * dataset["target_scale"] + dataset["target_mean"]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
