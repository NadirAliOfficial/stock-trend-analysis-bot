# IBKR Stock Trend Analysis Bot

An automated stock analysis and trading bot for Interactive Brokers that combines multi-timeframe momentum scoring, EMA/RSI technical analysis, and Facebook Prophet ML forecasting to identify and trade high-probability setups.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![IBKR](https://img.shields.io/badge/Interactive%20Brokers-API-red?style=flat)

## How It Works

1. Loads stock symbols from `listStocksAll_SCREENED.csv`
2. Fetches historical data across 6 timeframes (5Y, 1Y, 6M, 1M, 1W, 1D) via IBKR API
3. Scores each stock using EMA crossover + RSI trend filter
4. Runs Prophet ML forecast on top candidates
5. Places bracket orders on IBKR for high-score stocks
6. Saves results to `ibkr_trading_results.csv`

## Timeframes Analyzed

| Duration | Bar Size |
|----------|----------|
| 5 Years  | 1 day    |
| 1 Year   | 1 day    |
| 6 Months | 1 day    |
| 1 Month  | 1 hour   |
| 1 Week   | 5 mins   |
| 1 Day    | 5 mins   |

## Requirements

- IBKR TWS or IB Gateway running on port 7497
- `listStocksAll_SCREENED.csv` with a `Symbol` column

```bash
pip install ib_insync yfinance prophet pandas numpy
```

## Usage

```bash
python bot.py
```

Results are saved to `ibkr_trading_results.csv`.

## Notes

- Uses EMA (12/26) crossover for trend direction
- RSI for overbought/oversold filtering
- Prophet forecasts next-session price target
- Halal-compliant: no futures or options
- Real-time bracket orders with stop-loss and take-profit

## License

MIT


