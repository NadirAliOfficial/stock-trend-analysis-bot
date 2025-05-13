import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
from ib_insync import *
import warnings

warnings.filterwarnings("ignore")

# Connect to IBKR TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=12)

df_symbols = pd.read_csv('listStocksAll_SCREENED.csv')
symbols = df_symbols['stock'].dropna().tolist()


def create_contract(symbol):
    return Stock(symbol, 'SMART', 'USD')

def analyze_and_trade(symbol):
    try:
        print(f"Processing: {symbol}")
        df = yf.download(symbol, period='10d', interval='5m')
        if df.empty:
            return {'symbol': symbol, 'status': 'No Data'}

        df = df.reset_index()[['Datetime', 'Close']]
        df.columns = ['ds', 'y']

        # ML Forecasting with Prophet
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=12, freq='5min')
        forecast = m.predict(future)
        predicted_gain = (forecast.iloc[-1]['yhat'] - df['y'].iloc[-1]) / df['y'].iloc[-1] * 100

        # Trend Filter
        df['ema_fast'] = df['y'].ewm(span=12).mean()
        df['ema_slow'] = df['y'].ewm(span=26).mean()
        delta = df['y'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        score = 0
        if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]: score += 1
        if df['rsi'].iloc[-1] > 50: score += 1
        if predicted_gain > 0.1: score += 1

        if score >= 2:
            contract = create_contract(symbol)
            ib.qualifyContracts(contract)
            price = df['y'].iloc[-1]
            limit_price = round(price * 0.999, 2)
            take_profit = round(limit_price * 1.001, 2)
            stop_loss = round(limit_price * 0.997, 2)
            bracket = ib.bracketOrder('BUY', 1, limit_price, take_profit, stop_loss)
            for o in bracket:
                o.transmit = True
                ib.placeOrder(contract, o)
            return {'symbol': symbol, 'score': score, 'gain': round(predicted_gain, 2), 'order': 'Placed'}
        else:
            return {'symbol': symbol, 'score': score, 'gain': round(predicted_gain, 2), 'order': 'Skipped'}
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

# Run analysis
results = [analyze_and_trade(sym) for sym in symbols]
df_results = pd.DataFrame(results)
df_results.to_csv('ibkr_trading_results.csv', index=False)
print(df_results)
