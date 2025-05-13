
# IBKR Auto-Trading Bot

## Setup Instructions

1. Clone the repo or move the script into a folder  
2. Add your `listStocksAll_SCREENED.csv` with `Symbol` column

3. Install dependencies:
```bash
pip install ib_insync yfinance prophet pandas numpy
````

4. Start IBKR TWS or Gateway (ensure port 7497 is open)

5. Run the bot:

```bash
python bot.py
```

## Notes

* Uses EMA + RSI for trend filtering
* Forecasts price with Prophet (ML)
* Places IBKR bracket orders for high-score stocks
* Output saved to `ibkr_trading_results.csv`

✅ Halal-compliant (no futures)
✅ Real-time decision making
✅ CSV is editable for stock symbols
