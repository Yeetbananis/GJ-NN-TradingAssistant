from pyexpat import model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pytz
import shap
import argparse
import sys
from sklearn.cluster import DBSCAN
import traceback
import ta

# =====================================================================================
# ALL-IN-ONE SCRIPT V3 - FINAL FOOLPROOF VERSION WITH UNROLLED LSTM FOR SHAP
# =====================================================================================

# --- Part 1: Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = PROJECT_ROOT / "data/labels/manual_labels.csv"

# --- Default Path Definitions (will be set properly in the main block) ---
MODEL_SAVE_PATH = None
SCALER_SAVE_PATH = None
PROCESSED_DATA_DIR = None

# Model Hyperparameters
SEQUENCE_LENGTH = 24
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
# NOTE: This unrolled model uses 1 layer for simplicity and SHAP compatibility
NUM_LAYERS = 1
DROPOUT = 0 # No dropout for a single layer LSTM

# --- Part 2: All Necessary Functions and Classes ---

def load_and_preprocess_data(period_1h="5d", period_5m="3d"):
    """
    This function unconditionally downloads fresh 5m and 1h data from yfinance
    every time it is called. It also robustly handles potential data format
    errors from the API.
    """
    print("Downloading fresh 5m and 1h data for inference...")
    try:
        # Step 1: Unconditionally download the latest data every time.
        df_1h = yf.download("GBPJPY=X", period=period_1h, interval="1h", auto_adjust=False)
        df_5m = yf.download("GBPJPY=X", period=period_5m, interval="5m", auto_adjust=False)

        if df_1h.empty or df_5m.empty:
            print("ERROR: yfinance returned an empty dataframe. Data may be unavailable.")
            return None, None
            
    except Exception as e:
        print(f"ERROR: Could not download data from yfinance: {e}")
        return None, None
    
    processed_dfs = []
    dataframes_to_process = {"1h": df_1h, "5m": df_5m}

    for name, df_original in dataframes_to_process.items():
        df_clean = df_original[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_clean.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Step 2: Fix the DateParseError.
        # The 'errors='coerce'' argument will turn any unparseable rows (like the bad "Ticker" row)
        # into NaT (Not a Time), which are then immediately removed by the subsequent dropna().
        df_clean.index = pd.to_datetime(df_clean.index, utc=True, errors='coerce')
        df_clean.dropna(inplace=True)

        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        processed_dfs.append(df_clean)
    
    print("[OK] Data processed successfully.")
    return processed_dfs[0], processed_dfs[1]

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def find_clustered_levels(historical_df, eps=0.20, min_samples=3):
    """
    Analyzes long-term historical data to find major support and resistance zones
    using a vectorized method and price clustering.
    """
    print("Finding major S/R zones from historical data...")
    
    # --- NEW: Vectorized method to find swing points ---
    # This is much faster and avoids the looping error.
    df = historical_df.copy() # Use a copy to avoid modifying the original data
    
    # Find Swing Highs (a high with two lower highs on each side)
    is_swing_high = (df['High'] > df['High'].shift(1)) & \
                    (df['High'] > df['High'].shift(2)) & \
                    (df['High'] > df['High'].shift(-1)) & \
                    (df['High'] > df['High'].shift(-2))
    swing_high_prices = df['High'][is_swing_high]

    # Find Swing Lows (a low with two lower lows on each side)
    is_swing_low = (df['Low'] < df['Low'].shift(1)) & \
                   (df['Low'] < df['Low'].shift(2)) & \
                   (df['Low'] < df['Low'].shift(-1)) & \
                   (df['Low'] < df['Low'].shift(-2))
    swing_low_prices = df['Low'][is_swing_low]
    
    all_swing_prices = pd.concat([swing_high_prices, swing_low_prices]).dropna().values.flatten().tolist()
    # --- END: Vectorized method ---

    if not all_swing_prices:
        return pd.DataFrame(columns=['level', 'strength'])

    # Use DBSCAN to cluster the prices
    prices_reshaped = np.array(all_swing_prices).reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(prices_reshaped)

    # Process the clusters to define zones
    df_clusters = pd.DataFrame({'price': all_swing_prices, 'label': db.labels_})
    df_clusters = df_clusters[df_clusters['label'] != -1] # Filter out noise

    if df_clusters.empty:
        return pd.DataFrame(columns=['level', 'strength'])

    # Calculate the midpoint and strength of each zone
    zones = df_clusters.groupby('label')['price'].agg(['mean', 'count'])
    zones.rename(columns={'mean': 'level', 'count': 'strength'}, inplace=True)
    
    return zones.sort_values(by='level', ascending=True)\
    

def find_volume_levels(data_period="30d", interval="5m", bin_size_pips=5):
    """
    --- FINAL VERSION ---
    Calculates POC using a fallback to "Price Action Volume" if yfinance returns 0 for real volume.
    """
    print("Calculating multi-timeframe Volume POCs...")
    try:
        df_poc = yf.download("GBPJPY=X", period=data_period, interval=interval, auto_adjust=False, progress=False)
        if df_poc.empty:
            return {}

        if isinstance(df_poc.columns, pd.MultiIndex):
            df_poc.columns = df_poc.columns.get_level_values(0)
        df_poc.columns = [col.lower() for col in df_poc.columns]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_poc[col] = pd.to_numeric(df_poc[col], errors='coerce')
        df_poc.dropna(inplace=True)

        # --- THE CRITICAL FIX: DETECT ZERO VOLUME AND CREATE A PROXY ---
        if df_poc['volume'].sum() == 0:
            print("  -> Detected zero volume from yfinance. Calculating 'Price Action Volume' as a proxy.")
            # Calculate the price range of each candle
            price_range = df_poc['high'] - df_poc['low']
            # Calculate a moving average of the price range
            avg_range = price_range.rolling(window=20, min_periods=1).mean()
            # Create a "volume" proxy where higher price action equals higher volume
            df_poc['volume'] = price_range * avg_range
            df_poc['volume'] = df_poc['volume'].fillna(0)


        # Helper function (no changes needed here from the debug version)
        def get_poc_details(df_slice, timeframe_name):
            df_slice = df_slice.copy()
            if df_slice.empty: return None

            pip_value = 0.01
            bin_size = bin_size_pips * pip_value
            min_price, max_price = df_slice['low'].min(), df_slice['high'].max()
            if pd.isna(min_price) or pd.isna(max_price): return None

            price_bins = np.arange(min_price, max_price + bin_size, bin_size)
            if len(price_bins) < 2: return None

            df_slice['price_bin'] = pd.cut(df_slice['close'], bins=price_bins, right=False)
            volume_profile = df_slice.groupby('price_bin', observed=False)['volume'].sum()

            if volume_profile.empty or volume_profile.sum() == 0: return None

            poc_interval = volume_profile.idxmax()
            poc_price = poc_interval.mid
            poc_volume = volume_profile.max()
            profile_avg_volume = volume_profile.mean()

            return {'price': poc_price, 'volume': poc_volume, 'avg_volume': profile_avg_volume}

        candles_per_day = (24 * 60) // 5
        poc_monthly = get_poc_details(df_poc, "Monthly")
        poc_weekly = get_poc_details(df_poc.tail(7 * candles_per_day), "Weekly")
        poc_daily = get_poc_details(df_poc.tail(1 * candles_per_day), "Daily")

        return {
            'Monthly POC': poc_monthly,
            'Weekly POC': poc_weekly,
            'Daily POC': poc_daily
        }
    except Exception as e:
        print(f"  -> ERROR: An unhandled exception occurred in find_volume_levels: {e}")
        print(traceback.format_exc())
        return {}
    
def add_key_level_features(df_5m, df_1h, return_all_levels=False):
    print("Engineering 'Key Level' features...")
    
    # --- 1. Get Long-Term Price-Based Levels ---
    # (This section is the same as before)
    df_daily_longterm = yf.download("GBPJPY=X", period="2y", interval="1d", auto_adjust=False, progress=False)
    df_daily_longterm.columns = df_daily_longterm.columns.get_level_values(0)
    major_zones = find_clustered_levels(df_daily_longterm)
    long_term_levels = major_zones['level'].tolist() if not major_zones.empty else []
    last_7_days = df_daily_longterm.tail(7)
    last_30_days = df_daily_longterm.tail(30)
    short_term_levels = [
        last_7_days['High'].max(), last_7_days['Low'].min(),
        last_30_days['High'].max(), last_30_days['Low'].min()
    ]

    # --- 2. NEW: Get all Volume-Based Levels ---
    poc_levels_dict = find_volume_levels()
    poc_values = [v['price'] for v in poc_levels_dict.values() if v and v.get('price') is not None]

    # --- 3. Create a master list of ALL candidate levels ---
    all_levels = sorted(list(set(long_term_levels + short_term_levels + poc_values)))
    cleaned_price_levels = []
    if all_levels:
        cleaned_price_levels.append(all_levels[0])
        for i in range(1, len(all_levels)):
            if abs(all_levels[i] - cleaned_price_levels[-1]) > 0.25:
                cleaned_price_levels.append(all_levels[i])
    
    # --- 4. The rest of the function continues, using the master list ---
    # (The logic for feature engineering and finding the single closest S/R is the same)
    df = df_5m.copy()
    df['atr_14'] = calculate_atr(df, period=14)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['returns'] = df['close'].pct_change()
    pst = pytz.timezone('America/Los_Angeles')
    df.index = df.index.tz_convert(pst)
    df['hour'] = df.index.hour
    df['is_trading_session'] = ((df['hour'] >= 21) | (df['hour'] < 1)).astype(int)
    
    df_daily_5m = df.resample('D').agg({'open':'first','high':'max','low':'min','close':'last'})
    df['pdh'] = df_daily_5m['high'].shift(1).reindex(df.index, method='ffill')
    df['pdl'] = df_daily_5m['low'].shift(1).reindex(df.index, method='ffill')
    df['session_high'] = df['high'].rolling(window=288).max().shift(1)
    df['session_low'] = df['low'].rolling(window=288).min().shift(1)
    
    df['res_1_price'] = np.nan
    df['sup_1_price'] = np.nan
    df['res_1_reason'] = 'N/A'
    df['sup_1_reason'] = 'N/A'

    prices = df['close'].values
    res_prices = np.full_like(prices, np.nan)
    sup_prices = np.full_like(prices, np.nan)
    res_reasons = np.full(len(df), 'N/A', dtype=object)
    sup_reasons = np.full(len(df), 'N/A', dtype=object)

    for i in range(len(df)):
        price = prices[i]
        price_res_list = [l for l in cleaned_price_levels if l > price]
        price_sup_list = [l for l in cleaned_price_levels if l < price]
        res_prices[i] = min(price_res_list) if price_res_list else np.nan
        sup_prices[i] = max(price_sup_list) if price_sup_list else np.nan

    df['res_1_price'] = res_prices
    df['sup_1_price'] = sup_prices
    
    df['dist_to_res_1'] = (df['res_1_price'] / df['close']) - 1
    df['dist_to_sup_1'] = (df['sup_1_price'] / df['close']) - 1
    
    final_df = df.ffill().dropna()
    if return_all_levels:
        return final_df, cleaned_price_levels, poc_levels_dict
    else:
        return final_df

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers # Should be 1 for this version
        # Use LSTMCell, which processes one timestep at a time. This is transparent to SHAP.
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        self.direction_head = nn.Linear(hidden_size, 1)
        self.quality_head = nn.Linear(hidden_size, 5)
        self.reward_ratio_head = nn.Linear(hidden_size, 1)
        self.sl_prob_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Manually "unroll" the LSTM sequence
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)

        # Loop through each time step in the sequence
        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))
        
        # last_out is the final hidden state
        last_out = h_t
        
        direction_pred = torch.tanh(self.direction_head(last_out))
        quality_pred = self.quality_head(last_out)
        reward_ratio_pred = self.reward_ratio_head(last_out)
        sl_prob_pred = torch.sigmoid(self.sl_prob_head(last_out))
        
        return torch.cat([direction_pred, quality_pred, reward_ratio_pred, sl_prob_pred], dim=1)

def engineer_features(df_1h, df_5m):
    """
    Generate numeric features from 1h and 5m dataframes for training/inference.
    Returns a DataFrame with numeric columns only.
    """
    print("Engineering numeric features...")
    df = df_5m.copy()  # Use 5m data as primary, align with 1h where needed
    
    # Calculate technical indicators (all numeric)
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_5'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    
    # Add price-based features
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # Align 1h features (e.g., longer-term trend)
    df_1h_resampled = df_1h.resample('5min').ffill().reindex(df.index, method='ffill')
    df['ema_1h_50'] = ta.trend.EMAIndicator(df_1h_resampled['close'], window=50).ema_indicator()
    
    # Drop rows with NaN values from indicators
    df = df.dropna()
    
    if df.empty:
        print("ERROR: Feature DataFrame is empty after engineering.")
        sys.exit(1)
    
    # Verify all columns are numeric
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    if len(numeric_cols) != len(df.columns):
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        print(f"ERROR: Non-numeric columns detected in features: {non_numeric_cols}")
        sys.exit(1)
    
    print(f"Generated {len(numeric_cols)} numeric features.")
    return df

def do_training(args):
    # Load data
    df_1h, df_5m = load_and_preprocess_data(period_1h="730d", period_5m="60d")
    if df_1h is None or df_5m is None:
        print("ERROR: Could not load data for training.")
        sys.exit(1)

    # Feature Engineering
    print("Engineering features...")
    feature_df = engineer_features(df_1h, df_5m)

    # Normalize features
    cols_to_normalize = [col for col in feature_df.columns if feature_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    if not cols_to_normalize:
        print("ERROR: No numeric columns available for normalization.")
        sys.exit(1)

    feature_df = feature_df.copy()
    feature_df[cols_to_normalize] = feature_df[cols_to_normalize].replace(['N/A', ''], np.nan)
    rows_before = len(feature_df)
    feature_df = feature_df.dropna(subset=cols_to_normalize)
    rows_dropped = rows_before - len(feature_df)
    if rows_dropped > 0:
        print(f"  -> Dropped {rows_dropped} rows with missing values.")

    if feature_df.empty:
        print("ERROR: Feature DataFrame is empty after cleaning.")
        sys.exit(1)

    scaler = StandardScaler()
    print("Normalizing features...")
    try:
        feature_df[cols_to_normalize] = scaler.fit_transform(feature_df[cols_to_normalize])
    except Exception as e:
        print(f"ERROR: Normalization failed: {e}")
        sys.exit(1)

    # Save scaler
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # Load labels
    label_df = pd.read_csv(LABEL_FILE)
    valid_labels = label_df.dropna(subset=['entry_price']).copy()
    valid_labels['date_parsed'] = pd.to_datetime(valid_labels['date'])

    # Validate required label columns
    required_columns = ['setup_quality', 'max_reward_ratio', 'outcome_sl_hit']
    missing_columns = [col for col in required_columns if col not in valid_labels.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns in {LABEL_FILE}: {', '.join(missing_columns)}")
        print(f"Available columns: {', '.join(valid_labels.columns)}")
        print("Please ensure manual_labels.csv contains 'setup_quality', 'max_reward_ratio', and 'outcome_sl_hit'.")
        sys.exit(1)

    # Align features and labels
    aligned_labels_list = []
    for _, label_row in valid_labels.iterrows():
        label_date = label_row['date_parsed'].date()
        entry_price = label_row['entry_price']
        session_features = feature_df[(feature_df.index.date == label_date)]
        if session_features.empty:
            continue
        closest_entry_time = (session_features['close'] - entry_price).abs().idxmin()
        new_label_entry = label_row.drop(['date', 'date_parsed']).to_dict()
        new_label_entry['timestamp'] = closest_entry_time
        aligned_labels_list.append(new_label_entry)
    aligned_labels_df = pd.DataFrame(aligned_labels_list).set_index('timestamp')

    # Verify aligned labels have required columns
    if not all(col in aligned_labels_df.columns for col in required_columns):
        print(f"ERROR: Aligned labels missing required columns: {', '.join([col for col in required_columns if col not in aligned_labels_df.columns])}")
        sys.exit(1)

    # Prepare sequences
    X, y_dict = [], {col: [] for col in required_columns}
    for timestamp, row in aligned_labels_df.iterrows():
        try:
            end_idx = feature_df.index.get_loc(timestamp)
            start_idx = end_idx - SEQUENCE_LENGTH + 1
            if start_idx < 0:
                continue
            sequence = feature_df.iloc[start_idx:end_idx + 1][cols_to_normalize].values
            if sequence.shape[0] != SEQUENCE_LENGTH:
                continue
            X.append(sequence)
            for col in required_columns:
                y_dict[col].append(row[col])
        except Exception as e:
            print(f"Warning: Skipping label at {timestamp} due to error: {e}")
            continue

    if not X:
        print("ERROR: No valid sequences for training.")
        sys.exit(1)

    X = np.array(X)
    try:
        y = np.array([y_dict['setup_quality'], y_dict['max_reward_ratio'], y_dict['outcome_sl_hit']]).T
    except KeyError as e:
        print(f"ERROR: Failed to create label array: {e}")
        sys.exit(1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create dataset and dataloader
    class TradingDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define model
    class TradingLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(TradingLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 3)  # Predicting setup_quality, max_reward_ratio, outcome_sl_hit
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    model = TradingLSTM(input_size=len(cols_to_normalize), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Save processed data
    feature_df.to_pickle(PROCESSED_DATA_DIR / "feature_df.pkl")
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")


def do_inference(version_to_load=None):
    print("\n--- MODE: INFERENCE ---")

    # --- Load Specific or Latest Model Version ---
    models_dir = PROJECT_ROOT / "models"
    processed_data_base_dir = PROJECT_ROOT / "data"
    base_name = "gbpjpy_assistant"
    target_version = 0

    if version_to_load is not None:
        print(f"Attempting to load specified artifacts: VERSION {version_to_load}")
        model_path_candidate = models_dir / f"{base_name}_v{version_to_load}.pth"
        if model_path_candidate.exists():
            target_version = version_to_load
        else:
            print(f"❌ ERROR: Version {version_to_load} not found. Please check the /models directory.")
            return
    else:
        print("No version specified. Finding the latest available model...")
        version = 1
        latest_version = 0
        while True:
            model_path_candidate = models_dir / f"{base_name}_v{version}.pth"
            if model_path_candidate.exists():
                latest_version = version
                version += 1
            else:
                break
        
        if latest_version > 0:
            target_version = latest_version
        else:
            print("❌ ERROR: No trained models found. Please run 'train' mode first.")
            return

    MODEL_SAVE_PATH = models_dir / f"{base_name}_v{target_version}.pth"
    SCALER_SAVE_PATH = models_dir / f"scaler_v{target_version}.pkl"
    PROCESSED_DATA_DIR = processed_data_base_dir / f"processed_for_training_v{target_version}"
    
    print(f"[OK] Loading artifacts for VERSION {target_version}")
    print(f"   - Model:    {MODEL_SAVE_PATH.name}")
    print(f"   - Scaler:   {SCALER_SAVE_PATH.name}")
    print(f"   - Data Dir: {PROCESSED_DATA_DIR.name}")
    
    try:
        with open(SCALER_SAVE_PATH, 'rb') as f: scaler = pickle.load(f)
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
    except FileNotFoundError:
        print(f"ERROR: Could not find artifacts for Version {target_version}. The training run may have been incomplete.")
        return

    df_1h, df_5m = load_and_preprocess_data(period_1h="5d", period_5m="3d")
    if df_1h is None: return
    feature_df = add_key_level_features(df_5m, df_1h)
    
    # --- THIS BLOCK FIXES THE ERROR ---
    # Define the exact feature columns the model expects. This list must be maintained.
    # It seems your v3 model was trained with 18 features (including returns and volume_ma_20)
    model_feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 'atr_14', 'volume_ma_20',
        'returns', 'hour', 'is_trading_session', 'pdh', 'pdl', 'session_high',
        'session_low', 'res_1_price', 'sup_1_price', 'dist_to_res_1', 'dist_to_sup_1'
    ]
    
    # Determine the correct input_size from our fixed list
    input_size = len(model_feature_columns)
    
    # 2. Load Model with the CORRECT input size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get features with key levels data
    feature_df, all_price_levels, poc_levels_dict = add_key_level_features(df_5m, df_1h, return_all_levels=True)
    
    # 3. Prepare Input Sequence
    last_sequence_df = feature_df.tail(SEQUENCE_LENGTH).copy()
    
    # Ensure all required columns exist in the dataframe for filtering
    for col in model_feature_columns:
        if col not in last_sequence_df.columns:
            last_sequence_df[col] = 0 # Add missing columns and fill with 0
            
    # Filter the DataFrame to only include the features for the model
    model_input_df = last_sequence_df[model_feature_columns]
    # --- END OF FIX ---

    if len(model_input_df) < SEQUENCE_LENGTH:
        print(f"Not enough data for a full sequence. Need {SEQUENCE_LENGTH}, have {len(model_input_df)}.")
        return

    # Scale the filtered data
    cols_to_exclude = ['pdh', 'pdl', 'session_high', 'session_low', 'res_1_price', 'sup_1_price']
    cols_to_scale = [col for col in model_input_df.columns if col not in cols_to_exclude]
    sequence_scaled = model_input_df.copy()
    sequence_scaled[cols_to_scale] = scaler.transform(model_input_df[cols_to_scale])
    input_tensor = torch.tensor(sequence_scaled.values, dtype=torch.float32).unsqueeze(0).to(device)

    # Wrap the inference logic in a try-except to ensure clean exit
    try:
        # 4. Prediction
        prediction_tensor = model(input_tensor)
        
        # 5. SHAP Explanation
        print("Calculating SHAP values...")
        if X_train.shape[2] != input_size:
            print(f"  -> Warning: Shape of loaded X_train ({X_train.shape[2]}) does not match model input size ({input_size}). SHAP values may be unreliable.")
            # Attempt to fix by padding/truncating, though this is not ideal
            if X_train.shape[2] > input_size:
                X_train = X_train[:, :, :input_size]
            else:
                padding = np.zeros((X_train.shape[0], X_train.shape[1], input_size - X_train.shape[2]))
                X_train = np.concatenate([X_train, padding], axis=2)

        background = torch.tensor(X_train[np.random.choice(X_train.shape[0], min(20, len(X_train)), replace=False)], dtype=torch.float32).to(device)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor, check_additivity=False)
        
        def explain_output(output_name, shap_values_for_output, effect_positive="HIGHER", effect_negative="LOWER"):
            shap_last_step = shap_values_for_output[0, -1, :]
            total_abs_shap = np.sum(np.abs(shap_last_step))
            top_indices = np.argsort(np.abs(shap_last_step))[::-1][:5]
            print(f"\n--- Top 5 Factors Influencing {output_name} ---")
            for i in top_indices:
                feature_name = model_input_df.columns[i]
                shap_value = float(shap_last_step[i])
                percentage_contribution = (np.abs(shap_value) / total_abs_shap) * 100 if total_abs_shap > 0 else 0
                effect = effect_positive if shap_value > 0 else effect_negative
                print(f"  - {str(feature_name):<20} ({percentage_contribution:5.1f}%) | Pushed {output_name} {effect}")

        # 6. Unpack Predictions and Display
        dir_pred = prediction_tensor[:, 0].item()
        qual_logits = prediction_tensor[:, 1:6]
        rr_pred = prediction_tensor[:, 6].item()
        sl_prob = prediction_tensor[:, 7].item()
        predicted_star = torch.softmax(qual_logits, dim=1).argmax().item() + 1
        direction_text = "Buy" if dir_pred > 0 else "Sell"

        print("\n" + "="*75)
        print(f"--- Live Trading Assistant Prediction (Using Model v{target_version}) ---")
        print(f"Timestamp: {last_sequence_df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("\nPrediction:")
        print(f"  - Direction Bias:     {dir_pred:.2f} ({direction_text})")
        print(f"  - Predicted Quality:  {predicted_star} Stars")
        print(f"  - Predicted R:R:      {rr_pred:.2f}")
        print(f"  - SL Hit Probability: {sl_prob * 100:.2f}%")

        latest_data = last_sequence_df.iloc[-1]
        entry_price = latest_data['close']
        atr = latest_data['atr_14']
        
        print("\nSuggested Trade Parameters:")
        sl_reason_text = "N/A"
        if direction_text == "Buy":
            stop_loss_price = latest_data['sup_1_price'] - atr * 0.25
            risk_per_share = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk_per_share * rr_pred)
            sl_reason_text = f"Based on {latest_data.get('sup_1_reason', 'Price Structure')}"
        else: # Sell
            stop_loss_price = latest_data['res_1_price'] + atr * 0.25
            risk_per_share = stop_loss_price - entry_price
            take_profit_price = entry_price - (risk_per_share * rr_pred)
            sl_reason_text = f"Based on {latest_data.get('res_1_reason', 'Price Structure')}"
        tp_reason_text = f"Calculated from Predicted {rr_pred:.2f} R:R"

        print(f"  - Entry Price:        ~{entry_price:.5f}")
        print(f"  - Stop Loss:          ~{stop_loss_price:.5f} (Reason: {sl_reason_text})")
        print(f"  - Take Profit:        ~{take_profit_price:.5f} (Reason: {tp_reason_text})")
        
        explain_output("R:R", shap_values[:, :, :, 6], "HIGHER", "LOWER")
        explain_output("Direction", shap_values[:, :, :, 0], "more TOWARDS BUY", "more TOWARDS SELL")
        explain_output(f"Quality ({predicted_star} Stars)", shap_values[:, :, :, 1 + (predicted_star - 1)], "HIGHER", "LOWER")
        explain_output("SL Probability", shap_values[:, :, :, 7], "HIGHER", "LOWER")

        print("\nKey Levels:")
        current_price = latest_data['close']
        
        support_levels = [level for level in all_price_levels if level < current_price]
        print("  - Nearest 2 Price Supports:")
        for level in support_levels[-2:]:
            print(f"      - {level:.5f}")
        
        resistance_levels = [level for level in all_price_levels if level > current_price]
        print("  - Nearest 2 Price Resistances:")
        for level in resistance_levels[:2]:
            print(f"      - {level:.5f}")
        
        print("  - Volume Points of Control (POC):")
        if poc_levels_dict:
            for name, details in poc_levels_dict.items():
                if details and details.get('price') is not None:
                    price = details['price']
                    volume = details.get('volume', 0)
                    avg_volume = details.get('avg_volume', 0)
                    dominance = (volume / avg_volume) if avg_volume > 0 else 1.0
                    print(f"      - {name}: ~{price:.5f} (Dominance: {dominance:.1f}x Avg Vol)")
        
        print("="*75 + "\n")
        
        sys.exit(0)  # Explicitly exit with code 0 to indicate success
    except Exception as e:
        print(f"  -> Error in inference: {e}")
        print(traceback.format_exc())
        sys.exit(1)  # Exit with error code if an exception occurs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GBP/JPY Trading Assistant")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Available modes')

    train_parser = subparsers.add_parser('train', help="Train a new version of the model.")
    train_parser.add_argument('--model-path', type=str, default=None, help="Override default model save path.")
    train_parser.add_argument('--scaler-path', type=str, default=None, help="Override default scaler save path.")
    train_parser.add_argument('--data-dir', type=str, default=None, help="Override default processed data directory.")
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Set the learning rate.")
    train_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Set the batch size.")
    train_parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE, help="Set the LSTM hidden layer size.")
    train_parser.add_argument('--train-indices-path', type=str, default=None, help="Path to .npy file with training indices.")
    train_parser.add_argument('--val-indices-path', type=str, default=None, help="Path to .npy file with validation indices.")

    infer_parser = subparsers.add_parser('infer', help="Run inference using a trained model.")
    infer_parser.add_argument('-v', '--version', type=int, default=None, help="Specify model version to load. Defaults to the latest.")

    args = parser.parse_args()

    if args.mode == 'train':
        LEARNING_RATE = args.lr
        BATCH_SIZE = args.batch_size
        HIDDEN_SIZE = args.hidden_size

        if args.model_path:
            print("[OK] Overriding default save paths for a tuning run.")
            MODEL_SAVE_PATH = Path(args.model_path)
            SCALER_SAVE_PATH = Path(args.scaler_path)
            PROCESSED_DATA_DIR = Path(args.data_dir)
        else:
            models_dir = PROJECT_ROOT / "models"
            processed_data_base_dir = PROJECT_ROOT / "data"
            base_name = "gbpjpy_assistant"
            version = 1
            while True:
                model_path_candidate = models_dir / f"{base_name}_v{version}.pth"
                if not model_path_candidate.exists():
                    MODEL_SAVE_PATH = model_path_candidate
                    SCALER_SAVE_PATH = models_dir / f"scaler_v{version}.pkl"
                    PROCESSED_DATA_DIR = processed_data_base_dir / f"processed_for_training_v{version}"
                    print("="*60)
                    print(f"[OK] This training run will be saved as VERSION {version}")
                    print(f"   - Model:    {MODEL_SAVE_PATH.name}")
                    print(f"   - Scaler:   {SCALER_SAVE_PATH.name}")
                    print(f"   - Data Dir: {PROCESSED_DATA_DIR.name}")
                    print("="*60)
                    break
                version += 1

        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        do_training(args)

    elif args.mode == 'infer':
        do_inference(version_to_load=args.version)