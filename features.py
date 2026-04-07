import numpy as np
import pandas as pd


RAW_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def normalize_upload_frame(df):
    frame = df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    col_map = {}
    for col in frame.columns:
        clean = col.lower().strip()
        if clean in ("date", "time", "datetime"):
            col_map[col] = "Date"
        elif clean == "open":
            col_map[col] = "Open"
        elif clean == "high":
            col_map[col] = "High"
        elif clean == "low":
            col_map[col] = "Low"
        elif clean in ("close", "adj close", "adj_close", "adjclose"):
            col_map[col] = "Close"
        elif clean in ("volume", "vol"):
            col_map[col] = "Volume"

    frame = frame.rename(columns=col_map)
    missing = [col for col in ("Date", "Close") if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    return frame


def build_summary(prepared):
    return {
        "rows": int(len(prepared)),
        "features": int(len(prepared.columns) - 2),
        "start": prepared["Date"].iloc[0],
        "end": prepared["Date"].iloc[-1],
        "lastClose": float(prepared["Close"].iloc[-1]),
    }


def prepare_features(df):
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "Close" not in frame.columns:
        raise ValueError("Close column is required")

    close = frame["Close"].astype(float)
    frame["Open"] = frame["Open"].astype(float) if "Open" in frame.columns else close
    frame["High"] = frame["High"].astype(float) if "High" in frame.columns else close
    frame["Low"] = frame["Low"].astype(float) if "Low" in frame.columns else close
    frame["Volume"] = frame["Volume"].astype(float) if "Volume" in frame.columns else 1.0
    frame = frame.dropna(subset=["Close"]).reset_index(drop=True)

    close = frame["Close"].to_numpy(dtype=float)
    open_ = frame["Open"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)

    feat = pd.DataFrame({
        "Date": frame["Date"].dt.strftime("%Y-%m-%d"),
        "Close": close,
        "Open": open_,
        "High": high,
        "Low": low,
        "Volume": volume,
    })

    feat["LogReturn"] = np.concatenate([[np.nan], np.log((close[1:] + 1e-9) / (close[:-1] + 1e-9))])
    feat["Return3"] = pd.Series(close).pct_change(3).values
    feat["Return5"] = pd.Series(close).pct_change(5).values
    feat["RangePct"] = (high - low) / (close + 1e-9)
    feat["BodyPct"] = (close - open_) / (open_ + 1e-9)
    feat["GapPct"] = np.concatenate([[np.nan], (open_[1:] - close[:-1]) / (close[:-1] + 1e-9)])

    for window in (5, 10, 20, 50, 100):
        rolling = pd.Series(close).rolling(window)
        feat[f"SMA{window}"] = rolling.mean().values
        feat[f"DistSMA{window}"] = close / (feat[f"SMA{window}"] + 1e-9)

    for window in (5, 10, 20, 50):
        feat[f"EMA{window}"] = pd.Series(close).ewm(span=window, adjust=False).mean().values
        feat[f"DistEMA{window}"] = close / (feat[f"EMA{window}"] + 1e-9)

    feat["RSI14"] = _rsi(close, 14)
    feat["RSI7"] = _rsi(close, 7)

    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    feat["MACD"] = ema12 - ema26
    feat["MACDSignal"] = pd.Series(feat["MACD"]).ewm(span=9, adjust=False).mean().values
    feat["MACDHist"] = feat["MACD"] - feat["MACDSignal"]

    sma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    feat["BBUpper"] = sma20 + 2 * std20
    feat["BBLower"] = sma20 - 2 * std20
    feat["BBWidth"] = (feat["BBUpper"] - feat["BBLower"]) / (sma20 + 1e-9)
    feat["BBPos"] = (close - feat["BBLower"]) / (feat["BBUpper"] - feat["BBLower"] + 1e-9)

    feat["StochK"], feat["StochD"] = _stochastic(high, low, close, 14, 3)
    feat["ATR14"] = _atr(high, low, close, 14)
    feat["CCI20"] = _cci(high, low, close, 20)
    feat["WilliamsR"] = _williams_r(high, low, close, 14)
    feat["OBV"] = _obv(close, volume)

    feat["Momentum10"] = np.concatenate([np.full(10, np.nan), close[10:] - close[:-10]])
    feat["ROC10"] = np.concatenate([np.full(10, np.nan), (close[10:] - close[:-10]) / (close[:-10] + 1e-9) * 100])
    feat["Vol10"] = pd.Series(feat["LogReturn"]).rolling(10).std().values
    feat["Vol20"] = pd.Series(feat["LogReturn"]).rolling(20).std().values
    feat["VolRatio"] = feat["Vol10"] / (feat["Vol20"] + 1e-9)
    feat["VolumeChange"] = np.concatenate([[np.nan], np.diff(volume) / (volume[:-1] + 1e-9)])
    feat["VolSMA10"] = pd.Series(volume).rolling(10).mean().values
    feat["VolumeZ20"] = _rolling_zscore(volume, 20)

    for lag in (1, 2, 3, 5, 10):
        feat[f"CloseLag{lag}"] = pd.Series(close).shift(lag).values
        feat[f"RetLag{lag}"] = pd.Series(feat["LogReturn"]).shift(lag).values

    feat["HighLow20"] = close / (pd.Series(low).rolling(20).min().values + 1e-9)
    feat["HighHigh20"] = close / (pd.Series(high).rolling(20).max().values + 1e-9)
    feat["Trend10"] = pd.Series(close).pct_change(10).values
    feat["Trend20"] = pd.Series(close).pct_change(20).values

    dates_dt = frame["Date"]
    feat["DaySin"] = np.sin(2 * np.pi * dates_dt.dt.dayofweek.to_numpy() / 7)
    feat["DayCos"] = np.cos(2 * np.pi * dates_dt.dt.dayofweek.to_numpy() / 7)
    feat["MonthSin"] = np.sin(2 * np.pi * dates_dt.dt.month.to_numpy() / 12)
    feat["MonthCos"] = np.cos(2 * np.pi * dates_dt.dt.month.to_numpy() / 12)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(feat) < 140:
        raise ValueError("Not enough usable rows after feature engineering")
    return feat


def raw_history(df):
    return df[RAW_COLS].copy().reset_index(drop=True)


def next_business_date(date_value):
    last_date = pd.Timestamp(date_value)
    return pd.bdate_range(last_date, periods=2)[-1].strftime("%Y-%m-%d")


def extend_history(frame, next_close):
    raw = raw_history(frame)
    prev_close = float(raw["Close"].iloc[-1])
    recent_ret = raw["Close"].pct_change().tail(12).dropna()
    recent_vol = float(recent_ret.std()) if not recent_ret.empty else 0.01
    drift = (float(next_close) - prev_close) / (prev_close + 1e-9)
    band = prev_close * max(0.003, recent_vol, abs(drift) * 0.8)
    next_open = prev_close
    next_high = max(next_open, float(next_close)) + band
    next_low = max(1e-6, min(next_open, float(next_close)) - band)
    next_volume = float(raw["Volume"].tail(5).median()) if len(raw) >= 5 else float(raw["Volume"].iloc[-1])
    row = {
        "Date": next_business_date(raw["Date"].iloc[-1]),
        "Open": next_open,
        "High": next_high,
        "Low": next_low,
        "Close": float(next_close),
        "Volume": max(next_volume, 1.0),
    }
    return pd.concat([raw, pd.DataFrame([row])], ignore_index=True)


def _rolling_zscore(values, window):
    series = pd.Series(values)
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ((series - mean) / (std + 1e-9)).values


def _rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-9)
    return np.concatenate([[np.nan], 100 - 100 / (1 + rs)])


def _stochastic(high, low, close, k_period, d_period):
    size = len(close)
    k_vals = np.full(size, np.nan)
    for idx in range(k_period - 1, size):
        low_band = np.min(low[idx - k_period + 1:idx + 1])
        high_band = np.max(high[idx - k_period + 1:idx + 1])
        k_vals[idx] = 100 * (close[idx] - low_band) / (high_band - low_band + 1e-9)
    d_vals = pd.Series(k_vals).rolling(d_period).mean().values
    return k_vals, d_vals


def _atr(high, low, close, period):
    size = len(close)
    tr = np.full(size, np.nan)
    for idx in range(1, size):
        tr[idx] = max(high[idx] - low[idx], abs(high[idx] - close[idx - 1]), abs(low[idx] - close[idx - 1]))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def _cci(high, low, close, period):
    typical = (high + low + close) / 3
    size = len(typical)
    cci = np.full(size, np.nan)
    for idx in range(period - 1, size):
        window = typical[idx - period + 1:idx + 1]
        mean = np.mean(window)
        cci[idx] = (typical[idx] - mean) / (0.015 * np.mean(np.abs(window - mean)) + 1e-9)
    return cci


def _williams_r(high, low, close, period):
    size = len(close)
    wr = np.full(size, np.nan)
    for idx in range(period - 1, size):
        high_band = np.max(high[idx - period + 1:idx + 1])
        low_band = np.min(low[idx - period + 1:idx + 1])
        wr[idx] = -100 * (high_band - close[idx]) / (high_band - low_band + 1e-9)
    return wr


def _obv(close, volume):
    size = len(close)
    obv = np.zeros(size)
    for idx in range(1, size):
        if close[idx] > close[idx - 1]:
            obv[idx] = obv[idx - 1] + volume[idx]
        elif close[idx] < close[idx - 1]:
            obv[idx] = obv[idx - 1] - volume[idx]
        else:
            obv[idx] = obv[idx - 1]
    return obv
