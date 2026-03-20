# Why My Trading Bot Had an 85% Win Rate (and Why That Was the Problem)

Quantitative trading is often a race to find the "Holy Grail"—that one strategy that prints money while you sleep. Last week, I thought I’d found it. My new Transformer-based model was hitting an **85% win rate** on Binance data.

But as any seasoned quant will tell you: **If it looks too good to be true, it is.**

I decided to stop celebrating and start auditing. What I found was a masterclass in how ML models "cheat" when given the chance. 

### 1. The Zero-Day Leakage
The first red flag was the data split. My pipeline concatenated 50 different symbols into one massive sequence before splitting it into Training and Testing sets.
*   **The Cheat:** The Training set was swallowing 100% of the data for the first few symbols (BTC, ETH, etc.). When the backtester ran a "test" on the last 15% of BTC, the model wasn't predicting—it was *remembering*. It had already seen those exact candles during training.

### 2. Selection Bias (Peeking at the Future)
The most subtle flaw was in the entry logic. The script evaluated 64 potential entry prices and picked the "best" one.
*   **The Cheat:** The simulation only considered an entry "valid" if the price actually hit that level later in the day. This created a "Perfect Retry" mechanism. If the model’s first choice didn't fill, the backtester skipped to the next one that *did*. In the real world, you don't get a second chance once the market moves away.

### 3. Global Scaling Leakage
I was scaling my volatility and ATR features across the entire dataset before splitting.
*   **The Cheat:** The model knew the mean volatility of the *future* test data while it was still learning the training data. It was anticipating regime changes based on statistical shifts it shouldn't have known about yet.

### The Fix: Engineering for Reality
I’ve spent the last few days stripping away these "cheats":
✅ **Per-Symbol Splitting:** Data is now split by time *before* concatenation.
✅ **Fair Execution:** The model picks a price, and if it doesn't hit, it’s a "No Trade"—no second guesses.
✅ **Local Scaling:** Scalers are fit only on the training partition.

**The result?** The win rate dropped to a much more realistic—and profitable—**60%**. 

**The Lesson:** In AI trading, your biggest enemy isn't the market; it's your own validation logic. A "failed" 85% win rate is worth infinitely more than a fake one if it leads you to a robust 60%.

#MachineLearning #AlgorithmicTrading #Python #AI #Crypto #QuantitativeFinance
