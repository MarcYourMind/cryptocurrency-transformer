# Moving Beyond Accuracy: Why F1-Score is the King of Trading Metrics

If I told you my trading bot was 99% accurate, would you trust it? 
What if I told you the market only gives a "Buy" signal 1% of the time, and my bot just predicts "No Trade" for every single candle?

In trading, **Accuracy is a trap.** This is why I use **F1-Score and Calibration Curves** to measure my Transformer model’s success.

### The Imbalance Problem
In a sideways market, "No Trade" is often the right answer. If your dataset is 70% "No Trade" and 30% "Trade," a dumb model that always says "No" will have 70% accuracy—but $0 profit.

### The F1-Score Solution
The F1-score is the harmonic mean of **Precision** (how many of our "Buy" calls were right?) and **Recall** (how many of the total "Buy" opportunities did we actually catch?). 
*   High Precision = Fewer trades, but higher quality.
*   High Recall = More trades, but potentially more noise.
The F1-score forces the model to balance the two, ensuring we aren't just "playing it safe" or "gambling."

### The Calibration Curve
My model outputs a probability (e.g., "I am 62% sure this is a win"). For this to be useful, it needs to be **calibrated**.
If the model says there's a 60% probability, then in the backtest, we should see a win rate of exactly 60% for that confidence level. I use a **Calibration Curve** to visualize this. If the curve is a straight diagonal line, the model is "honest." If it’s curved, we might be overconfident or underconfident.

**The takeaway:** Don't chase accuracy. Chase a model that knows exactly how confident it is—and is honest about it.

#DataScience #TradingMetrics #F1Score #MachineLearning #CryptoTrading #ModelCalibration #QuantFinance
