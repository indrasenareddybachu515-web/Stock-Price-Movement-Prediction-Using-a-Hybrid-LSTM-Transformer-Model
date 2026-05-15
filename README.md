# Stock Price Movement Prediction Using a Hybrid LSTM–Transformer

**Module:** CMP-L016 — Deep Learning Applications
**Programme:** MSc Data Science, University of Roehampton London
**Author:** Indra Sena (Student ID: A00085075)
**Submission:** December 2025

This repository contains the final code, models and supporting material for the *Stock Price Movement Prediction* project. A hybrid Bidirectional LSTM + Transformer classifier is trained on S&P 500 daily OHLCV data (plus seven engineered technical indicators) to predict next-day directional movement (UP / DOWN), evaluated both on a held-out chronological split and on fresh data pulled live from Yahoo Finance, and deployed as a Gradio web application.

---

## Repository Layout

```
.
├── notebooks/
│   ├── Indra_Milestone2.ipynb          # Data prep, model definition, training, ablation
│   └── Indra_Milestone3.ipynb          # Out-of-sample testing + Gradio deployment
├── app.py                              # Standalone Gradio app (Hugging Face Spaces-ready)
├── model/
│   ├── best_model.pt                   # Best-validation checkpoint (PyTorch state_dict)
│   ├── final_model.pt                  # Final saved model
│   └── scaler.pkl                      # MinMax scaler fit on training data
├── results/
│   ├── training_history.json
│   ├── model_comparison.csv
│   └── figures/                        # All .png figures used in the report
├── report/
│   └── Indra_A00085075_Final_Report.pdf
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/IndraSena/stock-lstm-transformer.git
cd stock-lstm-transformer
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
yfinance
gradio
```

### 2. Reproduce training (Milestone 2)

Open `notebooks/Indra_Milestone2.ipynb` in Google Colab (recommended — it has a free GPU) or Jupyter. The notebook is designed to run top-to-bottom and produces all figures and the saved checkpoint without manual intervention.

- **Dataset:** [S&P 500 Index Stocks (Daily Updated)](https://www.kaggle.com/datasets/joebeachcapital/s-and-p500-index-stocks-daily-updated) by JoeBeachCapital on Kaggle.
- If the dataset is unavailable, the notebook falls back to synthetic OHLCV data so the pipeline still runs end-to-end.
- Runtime on a free Colab T4 is roughly 15–25 minutes including the ablation.

### 3. Evaluate on new data (Milestone 3)

Open `notebooks/Indra_Milestone3.ipynb`. This notebook:

1. Reproduces the architecture and loads the saved checkpoint.
2. Pulls the most recent 12 months of OHLCV for 24 representative tickers via `yfinance`.
3. Computes accuracy, AUC-ROC, F1, precision and recall, the confusion matrix and ROC/PR curves.
4. Reports per-ticker AUC so sector-level generalisation can be inspected.
5. Launches the Gradio web app.

### 4. Run the Gradio web app

```bash
python app.py
```

The app launches at `http://localhost:7860`. Pass `--share` to expose a public `gradio.live` link, e.g.:

```bash
python app.py --share
```

---

## What the Model Does

Given the last **30 trading days** of OHLCV plus seven technical indicators (RSI-14, MACD line and signal, Bollinger upper and lower bands, EMA-20, ATR-14), the model outputs:

- **Direction:** UP (next close > today's close) or DOWN.
- **Confidence:** Sigmoid probability in [0, 1].

### Architecture (summary)

```
Input  (B, 30, 12)
     │
     ▼  Linear (12 → 128) + LayerNorm + ReLU
     ▼  Bidirectional LSTM  ×2 layers, hidden = 128, dropout = 0.3
     ▼  Linear (256 → 128)
     ▼  Sinusoidal Positional Encoding (+)
     ▼  Transformer Encoder  ×2 layers, 4 heads, d_model = 128, d_ff = 256
     ▼  Mean pooling over time
     ▼  FC (128 → 64) + BatchNorm + ReLU + Dropout
     ▼  FC (64 → 1)
     ▼  Sigmoid  →  P(UP | X)
```

Parameters: ~920k trainable.

---

## Key Results

| Evaluation Split           | Accuracy | AUC-ROC | F1    |
|----------------------------|---------:|--------:|------:|
| Held-out S&P 500 test fold | 0.521    | 0.514   | 0.644 |
| New data (yfinance, agg.)  | 0.508    | 0.472   | 0.668 |

**Per-ticker AUC on new data** (top vs bottom):

- Strongest: **MRK 0.65, MSFT 0.63, AMZN 0.60, KO 0.59, COST 0.58, JNJ 0.58.**
- Weakest: WMT, MS, GS, PG (all below 0.50).

The model produces a small but positive predictive edge on mature, low-volatility tickers and collapses to near-random behaviour on high-beta tech, energy and financials, where the test window contained sharp regime changes. The honest reading is that daily directional prediction sits very close to the noise floor and the architecture itself is *not* the bottleneck — see the report for full discussion.

---

## Notes on Reproducibility

- Random seed: 42 (numpy and torch).
- Train/val/test split is **chronological per ticker** to prevent look-ahead leakage.
- The MinMax scaler is fit **only on the training portion**.
- Class imbalance is handled with `pos_weight` in `BCEWithLogitsLoss`, not by oversampling.

---

## Acknowledgements and AI Usage

In line with the University of Roehampton AI policy, I acknowledge that ChatGPT and Claude were used as coding assistants during this project (debugging PyTorch dataloaders, matplotlib styling and explaining edge cases of the early-stopping API). The analysis, interpretation, model design and the written report are my own work. All AI-suggested code was tested and verified before inclusion.

Dataset credit: JoeBeachCapital (Kaggle, S&P 500 Index Stocks Daily Updated). Yahoo Finance is used via the `yfinance` Python package for out-of-sample evaluation and live inference.

---

## Contact

Indra Sena · MSc Data Science · University of Roehampton London
Email: *sena.indra@roehampton.ac.uk*
