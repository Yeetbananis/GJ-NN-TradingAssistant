import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf

# -------------------------------
# CONFIG
# -------------------------------
SEQUENCE_LENGTH = 40
POST_ENTRY = 2
CANDLE_INTERVAL = '5m'
EMA_SHORT = 5
EMA_LONG = 20

# -------------------------------
# FUNCTIONS
# -------------------------------

def load_manual_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Entry_Time'])
    # Must contain: Trade_ID, Entry_Time, Entry_Price, TP, SL, Setup_Quality
    return df

def fetch_ohlcv(symbol, start_date, end_date, interval='5m'):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df = df.reset_index()
    df.rename(columns={'Datetime': 'Timestamp', 'Adj Close': 'Close'}, inplace=True)
    return df

def detect_major_zones(df):
    zones = []
    zone_id = 1
    # H1 zones
    df['H1'] = df['Timestamp'].dt.floor('H')
    h1_groups = df.groupby('H1')
    for _, group in h1_groups:
        zones.append({'Zone_ID': zone_id, 'Type': 'H1_High', 'Price_Level': group['High'].max(), 'Importance': 2}); zone_id+=1
        zones.append({'Zone_ID': zone_id, 'Type': 'H1_Low', 'Price_Level': group['Low'].min(), 'Importance': 2}); zone_id+=1
    # 15m zones
    df['M15'] = df['Timestamp'].dt.floor('15min')
    m15_groups = df.groupby('M15')
    for _, group in m15_groups:
        zones.append({'Zone_ID': zone_id, 'Type': 'M15_High', 'Price_Level': group['High'].max(), 'Importance': 1}); zone_id+=1
        zones.append({'Zone_ID': zone_id, 'Type': 'M15_Low', 'Price_Level': group['Low'].min(), 'Importance': 1}); zone_id+=1
    return pd.DataFrame(zones)

def compute_features(ohlcv, zones, entry_time, entry_price, tp, sl):
    mask = (ohlcv['Timestamp'] <= entry_time)
    window = ohlcv.loc[mask].tail(SEQUENCE_LENGTH + POST_ENTRY).copy()

    # EMAs
    window['EMA_short'] = window['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
    window['EMA_long'] = window['Close'].ewm(span=EMA_LONG, adjust=False).mean()
    window['EMA_Slope'] = window['EMA_short'] - window['EMA_long']

    # ATR for normalization
    window['H-L'] = window['High'] - window['Low']
    window['H-C'] = abs(window['High'] - window['Close'].shift(1))
    window['L-C'] = abs(window['Low'] - window['Close'].shift(1))
    window['TR'] = window[['H-L','H-C','L-C']].max(axis=1)
    ATR = window['TR'].rolling(14, min_periods=1).mean().iloc[-1]

    # Candle features
    candle_features = []
    for _, row in window.iterrows():
        distance_to_zone = np.min(np.abs(zones['Price_Level'] - row['Close']))
        nearest_zone = zones.iloc[np.argmin(np.abs(zones['Price_Level'] - row['Close']))]
        was_zone_swept = int((row['High'] >= nearest_zone['Price_Level']) and (row['Low'] <= nearest_zone['Price_Level']))
        sweep_size = (row['Close'] - nearest_zone['Price_Level']) / ATR if ATR != 0 else 0
        wick_body_ratio = ((row['High'] - row['Low']) / max(row['Close'] - row['Open'], 0.0001))
        candle_features.append({
            'Distance_to_Zone': distance_to_zone / ATR,
            'Was_Zone_Swept': was_zone_swept,
            'Sweep_Size': sweep_size,
            'Wick_Body_Ratio': wick_body_ratio,
            'EMA_Slope': row['EMA_Slope'],
            'Zone_Importance': nearest_zone['Importance']
        })

    # Entry features
    reward_to_risk = (tp - entry_price) / max(entry_price - sl, 0.0001)
    
    # Determine actual outcome: 1 = TP hit first, 0 = SL hit first
    future_mask = (ohlcv['Timestamp'] > entry_time)
    post_candles = ohlcv.loc[future_mask].head(50)  # check next 50 candles
    outcome_win = None
    actual_exit_price = None
    for _, c in post_candles.iterrows():
        if c['High'] >= tp:
            outcome_win = 1
            actual_exit_price = tp
            break
        elif c['Low'] <= sl:
            outcome_win = 0
            actual_exit_price = sl
            break
    # fallback if neither hit: use last candle close
    if outcome_win is None:
        outcome_win = 0
        actual_exit_price = post_candles['Close'].iloc[-1] if len(post_candles) > 0 else entry_price
    actual_rr = (actual_exit_price - entry_price) / max(entry_price - sl, 0.0001)

    entry_features = {
        'Entry_Price': entry_price,
        'TP': tp,
        'SL': sl,
        'Reward_to_Risk': reward_to_risk,
        'Outcome_Win': outcome_win,
        'Actual_RR': actual_rr
    }

    return candle_features, entry_features, ATR

def build_lstm_dataset(manual_csv_path, ohlcv_symbol='GBPJPY=X'):
    trades = load_manual_csv(manual_csv_path)
    start_date = trades['Entry_Time'].min() - timedelta(days=2)
    end_date = trades['Entry_Time'].max() + timedelta(days=1)

    ohlcv = fetch_ohlcv(ohlcv_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), CANDLE_INTERVAL)
    zones = detect_major_zones(ohlcv)

    dataset_rows = []

    for _, trade in trades.iterrows():
        candle_feats, entry_feats, ATR = compute_features(
            ohlcv,
            zones,
            trade['Entry_Time'],
            trade['Entry_Price'],
            trade['TP'],
            trade['SL']
        )

        flat_feats = {}
        for i, feat in enumerate(candle_feats):
            for k, v in feat.items():
                flat_feats[f't-{len(candle_feats)-i}_{k}'] = v

        row = flat_feats.copy()
        row.update(entry_feats)
        row['Setup_Quality'] = trade['Setup_Quality']
        row['Trade_ID'] = trade['Trade_ID']

        # Multi-head labels
        row['LSTM_Label_Setup_Quality'] = trade['Setup_Quality']
        row['LSTM_Label_Expected_RR'] = entry_feats['Reward_to_Risk']
        row['LSTM_Label_Win_Probability'] = entry_feats['Outcome_Win']

        dataset_rows.append(row)

    final_df = pd.DataFrame(dataset_rows)

    # Normalize numeric features
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Trade_ID', 'Setup_Quality', 'LSTM_Label_Setup_Quality',
                    'LSTM_Label_Expected_RR','LSTM_Label_Win_Probability']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    final_df[numeric_cols] = final_df[numeric_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    return final_df

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    manual_csv = "manual_trades.csv"
    lstm_ready_df = build_lstm_dataset(manual_csv)
    lstm_ready_df.to_csv("enhanced_manual_data.csv", index=False)
    print("LSTM-ready multi-task dataset saved as enhanced_manual_data.csv")
