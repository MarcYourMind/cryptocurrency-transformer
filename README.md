# Transformer-Based Side-Ways Trading Model

This folder contains a complete machine learning pipeline for predicting trading outcomes using **Volume Profiles** and **Transformers**. The model specifically targets **sideways (range-bound) market regimes**, attempting to predict whether a Long position will hit its take-profit target before its stop-loss.

## Overview

The pipeline uses 15-minute OHLCV data from Binance. It automatically detects periods where the price is moving sideways using linear regression and builds a high-dimensional feature representation based on the volume distribution (Volume Profile) within that window.

### Key Components

- **Data Fetching**: Automatically pulls the top 25 USDT trading pairs from Binance by volume.
- **Regime Detection**: Only trains/tests on windows where the price slope is near zero (sideways).
- **Volume Profile**: Transforms price and volume data into a normalized 64-bin histogram.
- **Contextual Features**: Integrates ATR, Volatility, Average Volume, and Trend Slope.
- **Transformer Model**: Uses a PyTorch Transformer Encoder to process the volume profile sequence, combined with a MLP head for binary classification.

## File Structure

| File | Description |
| :--- | :--- |
| `src/train_gpt.py` | **Main Entry Point**. Handles data download, preprocessing, dataset generation, model training, and saving artifacts. |
| `src/backtest.py` | Loads the trained model to run a detailed trading simulation on the test set, generating an equity curve and performance metrics. |
| `src/bot.py` | The live trading engine that schedules checks and manages trades. |
| `src/bingx_client.py` | BingX API wrapper for order execution and account management. |
| `src/strategy.py` | Encapsulates model inference and signal generation for live data. |
| `src/server.py` | FastAPI server that provides a REST API and serves the dashboard. |
| `static/` | Contains the `index.html` dashboard UI. |
| `requirements.txt` | List of Python dependencies (PyTorch, CCXT, FastAPI, etc.). |
| `data/` | Directory containing downloaded CSV market data (auto-generated). |
| `results/` | Directory containing the trained model `best_model.pth`, `scaler.pkl`, and visualization plots. |

## Live Trading Bot

The pipeline includes a fully functional live trading bot that can interact with the **BingX** exchange.

### Features
- **Scheduling**: Checks for new signals on every 15m candle close and manages open trades every 5m.
- **Risk Management**: Automatically calculates position sizes based on a percentage of your balance.
- **Web Dashboard**: A premium, real-time dashboard to monitor bot status, active trades, and history.
- **Dry Run Mode**: Safely test the bot without live orders.

### Deployment

#### 1. Configure Environment
Copy the example environment file and fill in your BingX API keys:
```bash
cp .env.example .env
```
Edit `.env` to set:
- `BINGX_API_KEY`: Your BingX API Key
- `BINGX_SECRET_KEY`: Your BingX API Secret
- `DRY_RUN`: Set to `False` to enable live trading

#### 2. Start the Server
Run the FastAPI server to bring up the dashboard and API:
```bash
python -m src.server
```
*Note: This command starts the web server but **does not** automatically start the trading logic.*

#### 3. Start the Bot
1. Navigate to [http://localhost:8000](http://localhost:8000) in your web browser.
2. In the **Control Panel**, click the green **START BOT** button.
3. Verify that the status badge changes to **RUNNING**.

The bot is now active and will:
- Check for new trade signals on every 15-minute candle close.
- Monitor and manage active trades every 5 minutes.

## How to Run (Development)

### 1. Set Up Environment
It is recommended to use a virtual environment. To activate the existing `env` folder:

**Windows:**
```powershell
.\env\Scripts\activate
```

**Linux/macOS:**
```bash
source env/bin/activate
```

If you haven't created the environment yet, run:
```bash
python -m venv env
```

### 2. Install Dependencies
Ensure your environment is active, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script. This will download data (if not already present), process it, and train the Transformer.
```bash
python -m src.train_gpt
```
*Note: Training will automatically save the best model and a calibration curve in the `results/` folder.*

### 3. Run Backtest
Once training is complete, evaluate the model's performance in a simulated trading environment:
```bash
python -m src.backtest
```
This script generates:
- `results/results_{timestamp}.json`: Complete simulation metrics and equity data.
- `results/equity_curve.png`: A visual representation of account growth over the test period.
- `results/prediction_dist.png`: A histogram of the model's prediction confidence.

### 4. Create report

Auto-detection: Finds the latest results_*.json if no file is specified.
Quantstats Report: Generates a full HTML performance report (Metrics, Drawdowns, Monthly Returns, etc.).
Easy execution: Run it with `python -m src.create_report`.

## Strategy Details
- **Entry**: Last close price of a sideways window.
- **Stop Loss**: 1.0 * ATR.
- **Take Profit**: 1.0 * ATR (Risk-Reward 1:1).
- **Training Goal**: Predict the probability of hitting Take Profit before Stop Loss.
