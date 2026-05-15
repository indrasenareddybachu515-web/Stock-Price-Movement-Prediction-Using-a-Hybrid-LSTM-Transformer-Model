"""
app.py — Stock Price Movement Predictor (Gradio web app)
CMP-L016 Deep Learning Applications — Milestone 3
Hybrid Bidirectional LSTM + Transformer

Usage:
    pip install -r requirements.txt
    python app.py            # local at http://localhost:7860
    python app.py --share    # public gradio.live URL

Author: Indra Sena (A00085075), University of Roehampton London.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless rendering for Gradio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Configuration — must match Milestone 2 exactly
# ─────────────────────────────────────────────────────────────────────
class Config:
    OHLCV_COLS          = ["Open", "High", "Low", "Close", "Volume"]
    WINDOW_SIZE         = 30
    INPUT_DIM           = 12       # 5 OHLCV + 7 indicators
    LSTM_HIDDEN         = 128
    LSTM_LAYERS         = 2
    LSTM_DROPOUT        = 0.3
    N_HEADS             = 4
    TRANSFORMER_LAYERS  = 2
    TRANSFORMER_DIM     = 128
    TRANSFORMER_DFF     = 256
    TRANSFORMER_DROPOUT = 0.1
    FC_HIDDEN           = 64
    OUTPUT_DIM          = 1
    CHECKPOINT_PATH     = "model/best_model.pt"

cfg    = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────
# Feature engineering — identical to Milestone 2
# ─────────────────────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI-14, MACD, MACD signal, Bollinger Bands (20-day, 2σ), EMA-20, ATR-14."""
    close, high, low = df["Close"], df["High"], df["Low"]

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20

    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    return df


# ─────────────────────────────────────────────────────────────────────
# Model architecture (identical to Milestone 2)
# ─────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class HybridLSTMTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.INPUT_DIM, cfg.LSTM_HIDDEN),
            nn.LayerNorm(cfg.LSTM_HIDDEN),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size    = cfg.LSTM_HIDDEN,
            hidden_size   = cfg.LSTM_HIDDEN,
            num_layers    = cfg.LSTM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.LSTM_DROPOUT if cfg.LSTM_LAYERS > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(cfg.LSTM_HIDDEN * 2, cfg.TRANSFORMER_DIM)
        self.pos_enc   = PositionalEncoding(cfg.TRANSFORMER_DIM,
                                            cfg.TRANSFORMER_DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.TRANSFORMER_DIM,
            nhead           = cfg.N_HEADS,
            dim_feedforward = cfg.TRANSFORMER_DFF,
            dropout         = cfg.TRANSFORMER_DROPOUT,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=cfg.TRANSFORMER_LAYERS)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.TRANSFORMER_DIM, cfg.FC_HIDDEN),
            nn.BatchNorm1d(cfg.FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(cfg.LSTM_DROPOUT),
            nn.Linear(cfg.FC_HIDDEN, cfg.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.lstm_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────────────────────────────
def load_model() -> HybridLSTMTransformer:
    model = HybridLSTMTransformer(cfg).to(DEVICE)
    if os.path.exists(cfg.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=DEVICE))
        print(f"[OK] Loaded checkpoint from {cfg.CHECKPOINT_PATH}")
    else:
        print(f"[WARN] No checkpoint at {cfg.CHECKPOINT_PATH}; "
              f"running with random weights (demo mode).")
    model.eval()
    return model

model = load_model()


# ─────────────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────────────
def predict_stock_movement(ticker: str):
    """
    Download last 14 months of OHLCV for `ticker`, compute features,
    run the model, and return:
      - prediction label
      - confidence string
      - a matplotlib figure of recent close prices
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please enter a ticker symbol.", "", None

    try:
        import yfinance as yf
    except ImportError:
        return "yfinance is not installed.", "", None

    try:
        end   = datetime.today()
        start = end - timedelta(days=420)        # 14 months
        df    = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        return f"Download failed: {e}", "", None

    if df is None or df.empty or len(df) < 60:
        return f"No usable data returned for '{ticker}'.", "", None

    df = df.reset_index().rename(columns=str.title)
    # yfinance sometimes uses 'Adj Close' and multi-index columns; normalise:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in tup if c]).strip("_") for tup in df.columns]
    for need in cfg.OHLCV_COLS:
        if need not in df.columns:
            return f"Missing column '{need}' in downloaded data.", "", None

    # Feature engineering
    df = add_technical_indicators(df).dropna().reset_index(drop=True)
    feat_cols = (cfg.OHLCV_COLS
                 + ["RSI_14", "MACD", "MACD_Signal",
                    "BB_Upper", "BB_Lower", "EMA_20", "ATR_14"])
    features = df[feat_cols].values.astype(np.float32)

    if len(features) < cfg.WINDOW_SIZE:
        return f"Not enough rows after feature engineering ({len(features)}).", "", None

    # Min-max scale per channel on the available history (proxy for the
    # training scaler when one is not bundled with the deployment).
    f_min, f_max = features.min(0), features.max(0)
    span = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)
    features = (features - f_min) / span

    window = features[-cfg.WINDOW_SIZE:]
    x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)  # (1, 30, 12)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    pred_label = "UP ▲" if prob >= 0.5 else "DOWN ▼"
    confidence = f"{prob:.2%}" if prob >= 0.5 else f"{(1 - prob):.2%}"

    # Recent-price chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    recent = df.tail(60)
    ax.plot(recent["Date"], recent["Close"], color="#0077B6", lw=1.6)
    ax.fill_between(recent["Date"], recent["Close"], recent["Close"].min() - 1,
                    color="#0077B6", alpha=0.12)
    ax.set_title(f"{ticker} — last 60 trading days", fontsize=11)
    ax.set_ylabel("Close ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    summary = (
        f"## {pred_label}\n\n"
        f"**Confidence:** {confidence}  \n"
        f"**Raw probability of UP:** {prob:.4f}  \n"
        f"**As of:** {recent['Date'].iloc[-1].strftime('%Y-%m-%d')}"
    )
    return summary, f"P(UP) = {prob:.4f}", fig


# ─────────────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────────────
def build_interface():
    import gradio as gr

    examples = [["AAPL"], ["MSFT"], ["GOOGL"], ["AMZN"], ["JPM"], ["JNJ"], ["MRK"]]

    with gr.Blocks(title="Stock Movement Predictor") as demo:
        gr.Markdown(
            "# 📈 Stock Price Movement Predictor\n"
            "Hybrid Bidirectional LSTM + Transformer trained on S&P 500 daily data. "
            "Enter a ticker; the model returns an UP/DOWN prediction for the next "
            "trading day with a confidence score. **Not financial advice.**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                ticker = gr.Textbox(label="Ticker symbol", value="AAPL")
                btn    = gr.Button("Predict", variant="primary")
                gr.Examples(examples=examples, inputs=[ticker])
            with gr.Column(scale=1):
                pred_md = gr.Markdown()
                raw     = gr.Textbox(label="Raw probability", interactive=False)
        chart = gr.Plot(label="Recent close prices")

        btn.click(predict_stock_movement, inputs=[ticker],
                  outputs=[pred_md, raw, chart])
        ticker.submit(predict_stock_movement, inputs=[ticker],
                      outputs=[pred_md, raw, chart])

        gr.Markdown(
            "---\n*Indra Sena · A00085075 · MSc Data Science, "
            "University of Roehampton London · CMP-L016 (2025).*"
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Expose a public gradio.live URL.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
