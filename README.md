# Adaptive Dual-Phase Hybrid PSO-GWO — Python Web App

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000
```

---

## File Structure

```
stock_app/
├── app.py              ← Flask backend (all models + optimisers)
├── requirements.txt    ← Dependencies
├── static/
│   └── index.html      ← Complete web UI (single file)
└── README.md
```

---

## What it does

1. **Fetch or upload** stock data (Yahoo Finance or your own CSV)
2. **Auto-computes 40+ technical indicators** (SMA, EMA, RSI, MACD, BB, ATR, CCI, OBV, etc.)
3. **Trains 14 models** with live progress cards:
   - Base: LSTM (MLP), GBDT, ELM
   - PSO-tuned: each base model
   - GWO-tuned: each base model
   - Hybrid PSO-GWO: each base model
   - ★ **Novelty: Adaptive Dual-Phase Hybrid PSO-GWO LSTM**

4. **Interactive Plotly charts** for every model:
   - Train / Test predictions + 30-day forecast
   - Residuals bar chart
   - Actual vs Predicted scatter
   - All-model comparison chart

5. **Novelty-specific visualisations:**
   - Phase 1 convergence (feature selection)
   - Phase 2 convergence (hyperparameter tuning)
   - Adaptive λ schedule (PSO ↔ GWO weight transition)
   - Selected feature chips

6. **Full leaderboard table** sorted by RMSE

---

## Models Explained

| # | Model | Optimizer |
|---|-------|-----------|
| 1 | Base LSTM | None (default) |
| 2 | Base GBDT | None (default) |
| 3 | Base ELM  | None (default) |
| 4–5 | LSTM + PSO/GWO | Hyperparameter tuning |
| 6–7 | ELM + PSO/GWO  | Hyperparameter tuning |
| 8–9 | GBDT + PSO/GWO | Hyperparameter tuning |
| 10–12 | All + PSO-GWO | Standard hybrid |
| 13 ★ | **Adaptive Dual-Phase** | Binary+Continuous adaptive hybrid |

---

## Novelty Algorithm

```
Phase 1 — Binary Adaptive Hybrid PSO-GWO:
  For each iteration t:
    λ(t) = 1 − (t/T)²          # adaptive weight
    new = λ·PSO_pos + (1−λ)·GWO_pos
    binary = rand < |tanh(new)| # V-shaped transfer function
    fitness = ELM_RMSE(selected_features) + 0.005·|mask|

Phase 2 — Continuous Adaptive Hybrid PSO-GWO:
  Optimise [nHidden, lr, dropout] using same λ schedule
  fitness = MLP_validation_RMSE(selected_features, hyperparams)

Final:
  Train full MLP-LSTM with Phase-1 features + Phase-2 hyperparams
  Generate 30-day recursive forecast
```

---

## Tips

- **Faster run**: Set Pop=10, Iterations=20 for quick demo
- **Better results**: Pop=30, Iterations=60, Seq=60
- **Indian stocks**: Use `.NS` suffix — e.g. `RELIANCE.NS`, `TCS.NS`
- **Crypto**: Works too — `BTC-USD`, `ETH-USD`
