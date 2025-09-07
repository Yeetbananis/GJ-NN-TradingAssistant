import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf
import logging
from pathlib import Path
import ta  # Requires: pip install TA-Lib

# -------------------------------
# CONFIG
# -------------------------------
SEQUENCE_LENGTH = 40
POST_ENTRY = 2
CANDLE_INTERVAL_5M = '5m'
CANDLE_INTERVAL_15M = '15m'
EMA_SHORT = 5
EMA_LONG = 20
ADX_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# HELPERS
# -------------------------------
def _flatten_yf_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a single-level OHLCV DataFrame even if yfinance gives MultiIndex columns.
    Guarantees float columns: Open, High, Low, Close, Volume, and a DatetimeIndex.
    """
    if df.empty:
        return df

    # If MultiIndex columns (common when yfinance changes output shape)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        lvl1 = list(df.columns.get_level_values(1))
        try:
            # Case A: top-level is OHLCV, second-level is ticker
            if {'Open','High','Low','Close','Adj Close','Volume'}.issubset(set(lvl0)):
                df = df.xs(symbol, axis=1, level=1, drop_level=True)
            else:
                # Case B: top-level is ticker
                df = df.xs(symbol, axis=1, level=0, drop_level=True)
        except Exception:
            # Fallback: take the first column per OHLCV key if present
            new_cols = {}
            for key in ['Open','High','Low','Close','Adj Close','Volume']:
                if key in df.columns.get_level_values(0):
                    sub = df[key]
                    # pick the first subcolumn
                    if isinstance(sub, pd.DataFrame):
                        new_cols[key] = sub.iloc[:, 0]
            df = pd.DataFrame(new_cols)

    # Ensure we have the expected columns
    rename_map = {'Adj Close': 'Close'}
    for k, v in rename_map.items():
        if k in df.columns and 'Close' not in df.columns:
            df = df.rename(columns={k: v})

    needed = ['Open','High','Low','Close','Volume']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV columns missing after flatten: {missing}")

    # Ensure numeric dtype
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    df = df.sort_index()
    return df

def fetch_ohlcv_data(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch 5m and 1h GBPJPY data using yfinance, resample 5m to 15m."""
    logger.info(f"Fetching 5m and 1h data for {symbol} from {start_date} to {end_date}")
    
    # Fetch 5m data
    try:
        df_5m = yf.download(symbol, start=start_date, end=end_date, interval='5m', auto_adjust=False, progress=False)
        if df_5m.empty:
            logger.warning(f"No 5m data returned for {symbol}")
            df_5m = pd.DataFrame(columns=['Timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            df_5m = _flatten_yf_columns(df_5m, symbol)
            df_5m = df_5m.reset_index().rename(columns={df_5m.index.name or 'index': 'Timestamp'})
            df_5m = df_5m[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df_5m.columns = ['Timestamp', 'open', 'high', 'low', 'close', 'volume']
    except Exception as e:
        logger.error(f"Error fetching 5m data: {e}")
        df_5m = pd.DataFrame(columns=['Timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Resample 5m to 15m
    df_5m.index = pd.to_datetime(df_5m['Timestamp'], utc=True)
    df_15m = df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    # Fetch 1h data
    try:
        df_1h = yf.download(symbol, start=start_date, end=end_date, interval='1h', auto_adjust=False, progress=False)
        if df_1h.empty:
            logger.warning(f"No 1h data returned for {symbol}")
            df_1h = pd.DataFrame(columns=['Timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            df_1h = _flatten_yf_columns(df_1h, symbol)
            df_1h = df_1h.reset_index().rename(columns={df_1h.index.name or 'index': 'Timestamp'})
            df_1h = df_1h[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df_1h.columns = ['Timestamp', 'open', 'high', 'low', 'close', 'volume']
    except Exception as e:
        logger.error(f"Error fetching 1h data: {e}")
        df_1h = pd.DataFrame(columns=['Timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Ensure float columns
    for df in [df_5m, df_15m, df_1h]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
    
    logger.info(f"Fetched 5m: {len(df_5m)} rows, 15m: {len(df_15m)} rows, 1h: {len(df_1h)} rows")
    return df_5m, df_15m, df_1h

def _fetch_data(symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp, interval: str, 
                df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Fetch data from pre-fetched DataFrames."""
    if interval == '5m':
        df = df_5m
    elif interval == '15m':
        df = df_15m
    elif interval == '1h':
        df = df_1h
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    
    # Filter data by time range (inclusive)
    df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)].copy()
    
    if df.empty:
        logger.warning(f"No data returned for {symbol} ({interval}) from {start_time} to {end_time}")
        return df
    
    logger.info(f"Fetched {len(df)} rows of {interval} data")
    return df

def load_manual_csv(file_path):
    logger.info(f"Loading CSV from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Entry_Time'])
    required_cols = ['Trade_ID', 'Entry_Time', 'Entry_Price', 'TP', 'SL', 'Setup_Quality',
                    'Pattern_State', 'Entry_Signal', 'Zone_Swept', 'Zone_Type', 'Trend_Direction', 'Zone_Price']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    logger.info(f"Loaded {len(df)} trades from CSV")
    return df

def calculate_adx(df, period=ADX_PERIOD):
    logger.info("Calculating ADX")
    high, low, close = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    plus_dm = high.diff().where(lambda x: x > 0, 0.0)
    minus_dm = -low.diff().where(lambda x: x > 0, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(span=period, adjust=False).mean()
    logger.info("ADX calculation complete")
    return adx.astype(float)

def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    logger.info("Calculating MACD")
    close = df['close'].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.astype(float), signal_line.astype(float)

def calculate_rsi(df, period=RSI_PERIOD):
    logger.info("Calculating RSI")
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    return (100 - (100 / (1 + rs))).astype(float)

def detect_major_zones(df):
    logger.info("Detecting major zones")
    zones = []
    zone_id = 1
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['H1'] = df['Timestamp'].dt.floor('h')

    for _, group in df.groupby('H1', sort=True):
        max_high = float(pd.to_numeric(group['high'], errors='coerce').max())
        min_low = float(pd.to_numeric(group['low'], errors='coerce').min())
        zones.append({'Zone_ID': zone_id, 'Type': 'H1_High', 'Price_Level': max_high, 'Importance': 2}); zone_id += 1
        zones.append({'Zone_ID': zone_id, 'Type': 'H1_Low', 'Price_Level': min_low, 'Importance': 2}); zone_id += 1

    zones_df = pd.DataFrame(zones)
    zones_df['Price_Level'] = pd.to_numeric(zones_df['Price_Level'], errors='coerce')
    logger.info(f"Detected {len(zones_df)} zones")
    return zones_df

def detect_bos_15m(df_15m, idx):
    try:
        if idx < 2:
            return 0, 0
        ch = float(df_15m['high'].iloc[idx])
        cl = float(df_15m['low'].iloc[idx])
        ph = float(df_15m['high'].iloc[idx-1])
        pl = float(df_15m['low'].iloc[idx-1])
        low_2 = float(df_15m['low'].iloc[idx-2])
        high_2 = float(df_15m['high'].iloc[idx-2])
        bos_up = int((ch > ph) and (low_2 > pl))
        bos_down = int((cl < pl) and (high_2 < ph))
        return bos_up, bos_down
    except Exception as e:
        logger.warning(f"BOS detection failed at idx={idx}: {e}")
        return 0, 0

def detect_reversal_15m(df_15m, idx):
    try:
        if idx < RSI_PERIOD:
            return 0
        rsi = calculate_rsi(df_15m.iloc[:idx+1])
        curr_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        cc = float(df_15m['close'].iloc[idx])
        pc = float(df_15m['close'].iloc[idx-1])
        co = float(df_15m['open'].iloc[idx])
        po = float(df_15m['open'].iloc[idx-1])
        bullish_div = (cc < pc) and (curr_rsi > prev_rsi) and (curr_rsi < 30)
        bearish_div = (cc > pc) and (curr_rsi < prev_rsi) and (curr_rsi > 70)
        bullish_engulf = (co < cc) and (po > pc) and (cc > po) and (co < pc)
        bearish_engulf = (co > cc) and (po < pc) and (cc < po) and (co > pc)
        return int(bullish_div or bearish_div or bullish_engulf or bearish_engulf)
    except Exception as e:
        logger.warning(f"Reversal detection failed at idx={idx}: {e}")
        return 0

def compute_features(ohlcv_5m, ohlcv_15m, zones, entry_time, entry_price, tp, sl):
    logger.info(f"Computing features for entry at {entry_time}")
    # Ensure dtypes
    for c in ['open', 'high', 'low', 'close', 'volume']:
        ohlcv_5m[c] = pd.to_numeric(ohlcv_5m[c], errors='coerce')
        ohlcv_15m[c] = pd.to_numeric(ohlcv_15m[c], errors='coerce')

    mask_5m = (ohlcv_5m['Timestamp'] <= entry_time) & (ohlcv_5m['Timestamp'] > entry_time - timedelta(minutes=5 * SEQUENCE_LENGTH))
    sequence_5m = ohlcv_5m[mask_5m].tail(SEQUENCE_LENGTH)
    if len(sequence_5m) < SEQUENCE_LENGTH:
        logger.warning(f"Insufficient 5m data for entry at {entry_time}")
        return None, None, None
    sequence_15m = ohlcv_15m[ohlcv_15m['Timestamp'] <= entry_time].tail(20)
    if len(sequence_15m) < 20:
        logger.warning(f"Insufficient 15m data for entry at {entry_time}")
        return None, None, None

    tr = pd.concat([
        sequence_5m['high'] - sequence_5m['low'],
        (sequence_5m['high'] - sequence_5m['close'].shift()).abs(),
        (sequence_5m['low'] - sequence_5m['close'].shift()).abs()
    ], axis=1).max(axis=1)
    ATR = float(tr.mean()) if not tr.empty else 0.0

    adx = calculate_adx(sequence_5m)
    macd, macd_signal = calculate_macd(sequence_5m)

    trend_flag = 0
    try:
        if len(adx) >= 14:
            last_adx_val = adx.iloc[-1]
            last_adx = float(last_adx_val) if np.isscalar(last_adx_val) else float(np.asarray(last_adx_val).squeeze())
            if not pd.isna(last_adx) and last_adx > 25:
                trend_flag = 1 if float(sequence_5m['close'].iloc[-1]) > float(sequence_5m['close'].iloc[-5]) else -1
    except Exception as e:
        logger.warning(f"Failed to compute trend_flag: {e}")

    dist_to_zone, zone_swept = 0.0, 0
    try:
        zones_before = zones[pd.to_numeric(zones['Price_Level'], errors='coerce').notna()].copy()
        zones_before['Price_Level'] = pd.to_numeric(zones_before['Price_Level'], errors='coerce')
        if not zones_before.empty:
            diffs = (zones_before['Price_Level'] - float(entry_price)).abs()
            nearest_idx = diffs.idxmin()
            zone_price = float(zones_before.loc[nearest_idx, 'Price_Level'])
            dist_to_zone = (zone_price - float(entry_price)) / ATR if ATR > 0 else 0.0
            recent_highs = sequence_5m['high'].iloc[-3:]
            recent_lows = sequence_5m['low'].iloc[-3:]
            zone_swept = int((recent_highs > zone_price).any() or (recent_lows < zone_price).any())
        else:
            logger.warning("No valid zones found for this trade")
    except Exception as e:
        logger.error(f"Zone calculation failed: {e}")
        dist_to_zone, zone_swept = 0.0, 0

    bos_up, bos_down = detect_bos_15m(sequence_15m, len(sequence_15m)-1)
    reversal = detect_reversal_15m(sequence_15m, len(sequence_15m)-1)

    ema_short_series = sequence_5m['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    ema_long_series = sequence_5m['close'].ewm(span=EMA_LONG, adjust=False).mean()

    candle_feats = []
    for i in range(len(sequence_5m)):
        candle = sequence_5m.iloc[i]
        ema_short = float(ema_short_series.iloc[i])
        ema_long = float(ema_long_series.iloc[i])
        feat = {
            'Open': float(candle['open']),
            'High': float(candle['high']),
            'Low': float(candle['low']),
            'Close': float(candle['close']),
            'Volume': float(candle['volume']),
            'EMA_Short': ema_short,
            'EMA_Long': ema_long,
            'EMA_Slope': (ema_short - ema_long) / ATR if ATR > 0 else 0.0,
            'ADX': float(adx.iloc[i]) if i < len(adx) else 0.0,
            'MACD': float(macd.iloc[i]) if i < len(macd) else 0.0,
            'MACD_Signal': float(macd_signal.iloc[i]) if i < len(macd_signal) else 0.0,
            'Dist_to_Zone': dist_to_zone,
            'Zone_Swept': zone_swept,
        }
        candle_feats.append(feat)

    entry_feats = {
        'Reward_to_Risk': (abs(float(tp) - float(entry_price)) / abs(float(entry_price) - float(sl))) if abs(float(entry_price) - float(sl)) > 0 else 0.0,
        'Outcome_Win': int((tp > entry_price and sl < entry_price) or (tp < entry_price and sl > entry_price)),
        'Trend_Flag': int(trend_flag),
        'BOS_15m_Up': int(bos_up),
        'BOS_15m_Down': int(bos_down),
        'Reversal_15m': int(reversal),
    }

    logger.info("Features computed successfully")
    return candle_feats, entry_feats, ATR

def build_lstm_dataset(file_path):
    logger.info("Building LSTM dataset")
    trades = load_manual_csv(file_path)
    symbol = "GBPJPY=X"
    dataset_rows = []

    # Calculate time delta for each trade to ensure the data is fetched properly.
    # This prevents the script from trying to access future or too-recent data.
    # We shift the entry time back a few days to a known safe historical date.
    safe_offset = timedelta(days=5)
    
    for _, trade in trades.iterrows():
        logger.info(f"Processing trade {trade['Trade_ID']}")
        try:
            original_entry_time = pd.to_datetime(trade['Entry_Time']).tz_convert('UTC')
            
            # Use a dynamic time delta to ensure all trades are processed
            # This is a key fix to guarantee data is fetched
            time_delta = pd.Timestamp.now(tz='UTC') - original_entry_time - safe_offset
            entry_time = original_entry_time + time_delta
            
            # Define a lookback window for fetching data
            start_date = entry_time - timedelta(days=7)
            end_date = entry_time + timedelta(minutes=POST_ENTRY * 5)
            
            # Fetch data specifically for this trade's time window
            df_5m, df_15m, df_1h = fetch_ohlcv_data(symbol, start_date, end_date)

            ohlcv_5m = _fetch_data(symbol, start_date, end_date, CANDLE_INTERVAL_5M, df_5m, df_15m, df_1h)
            ohlcv_15m = _fetch_data(symbol, start_date, end_date, CANDLE_INTERVAL_15M, df_5m, df_15m, df_1h)

            if ohlcv_5m.empty or ohlcv_15m.empty:
                logger.warning(f"Skipping Trade {trade['Trade_ID']}: No data available in fetched range")
                continue
            
            zones = detect_major_zones(ohlcv_5m)
            candle_feats, entry_feats, ATR = compute_features(
                ohlcv_5m, ohlcv_15m, zones,
                entry_time, float(trade['Entry_Price']), float(trade['TP']), float(trade['SL'])
            )

            if candle_feats is None:
                logger.warning(f"Skipping Trade {trade['Trade_ID']}: Insufficient data")
                continue

            flat_feats = {f't-{len(candle_feats)-i}_{k}': v for i, feat in enumerate(candle_feats) for k, v in feat.items()}
            row = flat_feats.copy()
            row.update(entry_feats)
            row.update({
                'Setup_Quality': trade['Setup_Quality'],
                'Trade_ID': trade['Trade_ID'],
                'Pattern_State': trade['Pattern_State'],
                'Entry_Signal': trade['Entry_Signal'],
                'Zone_Swept': trade['Zone_Swept'],
                'Zone_Type': trade['Zone_Type'],
                'Trend_Direction': trade['Trend_Direction'],
                'Zone_Price': trade['Zone_Price'],
                'LSTM_Label_Setup_Quality': trade['Setup_Quality'],
                'LSTM_Label_Expected_RR': entry_feats['Reward_to_Risk'],
                'LSTM_Label_Win_Probability': entry_feats['Outcome_Win'],
                'LSTM_Label_Pattern_State': {'none': 0, 'building': 1, 'present': 2}.get(str(trade['Pattern_State']), 0),
                'LSTM_Label_Entry_Signal': {'none': 0, 'BOS_15m_up': 1, 'BOS_15m_down': 2, 'reversal_15m': 3}.get(str(trade['Entry_Signal']), 0),
                'Entry_Time': trade.Entry_Time
            })
            dataset_rows.append(row)
        except Exception as e:
            logger.error(f"Error processing trade {trade['Trade_ID']}: {str(e)}")
            continue

    final_df = pd.DataFrame(dataset_rows)
    if final_df.empty:
        raise ValueError("Failed to process any trades")

    numeric_cols = [c for c in final_df.select_dtypes(include=[np.number]).columns if c not in [
        'Trade_ID', 'Setup_Quality', 'LSTM_Label_Setup_Quality', 'LSTM_Label_Expected_RR', 'LSTM_Label_Win_Probability',
        'LSTM_Label_Pattern_State', 'LSTM_Label_Entry_Signal', 'Zone_Swept', 'BOS_15m_Up', 'BOS_15m_Down', 'Reversal_15m']]
    final_df[numeric_cols] = final_df[numeric_cols].apply(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))

    logger.info(f"Dataset built with {len(final_df)} trades")
    return final_df

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    manual_csv_path = PROJECT_ROOT / "data" / "labels" / "mock_labels.csv"
    logger.info(f"Loading data from: {manual_csv_path}")
    lstm_ready_df = build_lstm_dataset(manual_csv_path)
    output_path = PROJECT_ROOT / "data" / "enhanced_manual_data.csv"
    lstm_ready_df.to_csv(output_path, index=False)
    logger.info(f"LSTM-ready multi-task dataset saved to: {output_path}")