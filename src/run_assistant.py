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

# =====================================================================================
# ALL-IN-ONE SCRIPT V3 - FINAL FOOLPROOF VERSION WITH UNROLLED LSTM FOR SHAP
# =====================================================================================

# --- Part 1: Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = PROJECT_ROOT / "data/labels/manual_labels.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models/gbpjpy_assistant_v1.pth"
SCALER_SAVE_PATH = PROJECT_ROOT / "models/scaler.pkl"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed_for_training"

# Model Hyperparameters
SEQUENCE_LENGTH = 24
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
# NOTE: This unrolled model uses 1 layer for simplicity and SHAP compatibility
NUM_LAYERS = 1
DROPOUT = 0 # No dropout for a single layer LSTM

# --- Part 2: All Necessary Functions and Classes ---

def load_and_preprocess_data(period_1h="2y", period_5m="59d"):
    # (This function is correct, included for completeness)
    print("Loading/Downloading raw data...")
    raw_data_dir = PROJECT_ROOT / "data/raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    path_1h = raw_data_dir / "GBPJPY=X_1h.csv"
    path_5m = raw_data_dir / "GBPJPY=X_5m.csv"
    try:
        if not path_1h.exists():
            df_1h = yf.download("GBPJPY=X", period=period_1h, interval="1h", auto_adjust=False); df_1h.to_csv(path_1h)
        else: df_1h = pd.read_csv(path_1h, index_col=0)
        if not path_5m.exists():
            df_5m = yf.download("GBPJPY=X", period=period_5m, interval="5m", auto_adjust=False); df_5m.to_csv(path_5m)
        else: df_5m = pd.read_csv(path_5m, index_col=0)
    except Exception as e:
        print(f"Error downloading or loading raw data: {e}"); return None, None
    processed_dfs = []
    for df_original in [df_1h, df_5m]:
        df_clean = df_original[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_clean.columns = ['open', 'high', 'low', 'close', 'volume']
        df_clean.index = pd.to_datetime(df_clean.index, utc=True)
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        processed_dfs.append(df_clean.dropna())
    print("Raw data processed successfully.")
    return processed_dfs[0], processed_dfs[1]

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def add_key_level_features(df_5m, df_1h):
    # (This function is correct, included for completeness)
    print("Engineering 'Key Level' features...")
    df = df_5m.copy()
    df['atr_14'] = calculate_atr(df, period=14)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['returns'] = df['close'].pct_change()
    pst = pytz.timezone('America/Los_Angeles')
    df.index = df.index.tz_convert(pst)
    df['hour'] = df.index.hour
    df['is_trading_session'] = ((df['hour'] >= 21) | (df['hour'] < 1)).astype(int)
    df_daily = df.resample('D').agg({'open':'first','high':'max','low':'min','close':'last'})
    df['pdh'] = df_daily['high'].shift(1).reindex(df.index, method='ffill')
    df['pdl'] = df_daily['low'].shift(1).reindex(df.index, method='ffill')
    df['session_high'] = df['high'].rolling(window=96).max().shift(1)
    df['session_low'] = df['low'].rolling(window=96).min().shift(1)
    df['res_1_price'] = np.nan
    df['sup_1_price'] = np.nan
    levels = df[['pdh', 'pdl', 'session_high', 'session_low']].values
    prices = df['close'].values
    res_prices = np.full_like(prices, np.nan)
    sup_prices = np.full_like(prices, np.nan)
    for i in range(len(df)):
        current_levels = sorted([l for l in levels[i] if pd.notna(l)])
        res_level = [l for l in current_levels if l > prices[i]]
        sup_level = [l for l in current_levels if l < prices[i]]
        if res_level: res_prices[i] = min(res_level)
        if sup_level: sup_prices[i] = max(sup_level)
    df['res_1_price'] = res_prices
    df['sup_1_price'] = sup_prices
    df['dist_to_res_1'] = (df['res_1_price'] / df['close']) - 1
    df['dist_to_sup_1'] = (df['sup_1_price'] / df['close']) - 1
    return df.dropna()

# *** THE FOOLPROOF FIX: Replace the MultiTaskLSTM class with this version ***
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

def do_training():
    print("\n--- MODE: TRAINING ---")
    # 1. Data Loading & Feature Engineering
    df_1h, df_5m = load_and_preprocess_data()
    if df_1h is None: return
    feature_df = add_key_level_features(df_5m, df_1h)
    label_df = pd.read_csv(LABEL_FILE)
    
    # 2. Label Processing
    valid_labels = label_df.dropna(subset=['entry_price']).copy()
    valid_labels = valid_labels[~valid_labels['date'].duplicated(keep='first')]
    valid_labels['date_parsed'] = pd.to_datetime(valid_labels['date'])
    aligned_labels_list = []
    for _, label_row in valid_labels.iterrows():
        label_date = label_row['date_parsed'].date()
        entry_price = label_row['entry_price']
        session_features = feature_df[(feature_df.index.date == label_date) & (feature_df['is_trading_session'] == 1)]
        if session_features.empty: continue
        closest_entry_time = (session_features['close'] - entry_price).abs().idxmin()
        new_label_entry = label_row.drop(['date', 'date_parsed']).to_dict()
        new_label_entry['timestamp'] = closest_entry_time
        aligned_labels_list.append(new_label_entry)
    aligned_labels_df = pd.DataFrame(aligned_labels_list).set_index('timestamp')
    print(f"Successfully aligned {len(aligned_labels_df)} labels.")

    # 3. Normalization and Scaler Saving
    cols_to_exclude = ['pdh', 'pdl', 'session_high', 'session_low', 'res_1_price', 'sup_1_price']
    cols_to_normalize = [col for col in feature_df.columns if col not in cols_to_exclude]
    scaler = StandardScaler()
    feature_df[cols_to_normalize] = scaler.fit_transform(feature_df[cols_to_normalize])
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_SAVE_PATH, 'wb') as f: pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    # 4. Sequence Creation & Data Splitting
    X, y_dict = [], {col: [] for col in aligned_labels_df.columns}
    for timestamp, row in aligned_labels_df.iterrows():
        try:
            end_idx = feature_df.index.get_loc(timestamp)
            start_idx = end_idx - SEQUENCE_LENGTH + 1
            if start_idx < 0: continue
            X.append(feature_df.iloc[start_idx:end_idx + 1].values)
            for col, val in row.items(): y_dict[col].append(val)
        except KeyError: continue
    X, y = np.array(X), {k: np.array(v) for k, v in y_dict.items()}
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / 'X_train.npy', X)
    
    X_train, X_val, y_train_idx, y_val_idx = train_test_split(X, np.arange(len(X)), test_size=0.2, random_state=42)
    y_train = {key: val[y_train_idx] for key, val in y.items()}
    y_val = {key: val[y_val_idx] for key, val in y.items()}
    
    class TradingDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X = torch.tensor(X_data, dtype=torch.float32)
            self.y_dir = torch.tensor(y_data['trade_direction'], dtype=torch.float32).unsqueeze(1)
            self.y_qual = torch.tensor(y_data['setup_quality'] - 1, dtype=torch.long)
            self.y_rr = torch.tensor(y_data['max_reward_ratio'], dtype=torch.float32).unsqueeze(1)
            self.y_sl = torch.tensor(y_data['outcome_sl_hit'], dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], (self.y_dir[idx], self.y_qual[idx], self.y_rr[idx], self.y_sl[idx])

    train_ds = TradingDataset(X_train, y_train)
    val_ds = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskLSTM(X.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    loss_fns = {'direction': nn.MSELoss(), 'quality': nn.CrossEntropyLoss(), 'reward_ratio': nn.MSELoss(), 'sl_probability': nn.BCELoss()}
    loss_weights = {'direction': 0.5, 'quality': 1.5, 'reward_ratio': 1.0, 'sl_probability': 1.0}
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--- Starting Training on {device} ---")
    best_val_loss = float('inf')
    
    # ** CHANGE 3: Add counter for early stopping **
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        # (Training loop logic for one epoch remains the same)
        for features, labels_tuple in train_loader:
            features = features.to(device)
            preds = model(features)
            pred_dir, pred_qual, pred_rr, pred_sl = preds[:,0:1], preds[:,1:6], preds[:,6:7], preds[:,7:8]
            true_dir, true_qual, true_rr, true_sl = [lbl.to(device) for lbl in labels_tuple]
            loss = (loss_weights['direction'] * loss_fns['direction'](pred_dir, true_dir) +
                    loss_weights['quality'] * loss_fns['quality'](pred_qual, true_qual) +
                    loss_weights['reward_ratio'] * loss_fns['reward_ratio'](pred_rr, true_rr) +
                    loss_weights['sl_probability'] * loss_fns['sl_probability'](pred_sl, true_sl))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # (Validation loop logic remains mostly the same)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels_tuple in val_loader:
                features = features.to(device)
                preds = model(features)
                pred_dir, pred_qual, pred_rr, pred_sl = preds[:,0:1], preds[:,1:6], preds[:,6:7], preds[:,7:8]
                true_dir, true_qual, true_rr, true_sl = [lbl.to(device) for lbl in labels_tuple]
                loss = (loss_weights['direction'] * loss_fns['direction'](pred_dir, true_dir) +
                        loss_weights['quality'] * loss_fns['quality'](pred_qual, true_qual) +
                        loss_weights['reward_ratio'] * loss_fns['reward_ratio'](pred_rr, true_rr) +
                        loss_weights['sl_probability'] * loss_fns['sl_probability'](pred_sl, true_sl))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.4f}")
        
        # ** CHANGE 4: Implement the early stopping logic **
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved (Val Loss: {avg_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement.")
            break
            
    print("--- Training Complete ---")

def do_inference():
    print("\n--- MODE: INFERENCE ---")

    # --- NEW: Force redownload of raw data for inference ---
    print("Clearing old raw data to ensure freshness...")
    try:
        (PROJECT_ROOT / "data/raw/GBPJPY=X_1h.csv").unlink(missing_ok=True)
        (PROJECT_ROOT / "data/raw/GBPJPY=X_5m.csv").unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Could not delete old raw data files. {e}")
    
    # 1. Load artifacts
    try:
        with open(SCALER_SAVE_PATH, 'rb') as f: scaler = pickle.load(f)
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
    except FileNotFoundError:
        print(f"ERROR: Model/scaler not found. Please run 'train' mode first: python run_assistant.py train")
        return

    df_1h, df_5m = load_and_preprocess_data(period_1h="5d", period_5m="3d")
    if df_1h is None: return
    feature_df = add_key_level_features(df_5m, df_1h)
    
    # 2. Load Model
    input_size = len(feature_df.columns)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 3. Prepare Input Sequence
    last_sequence_df = feature_df.tail(SEQUENCE_LENGTH)
    if len(last_sequence_df) < SEQUENCE_LENGTH:
        print(f"Not enough data for a full sequence. Need {SEQUENCE_LENGTH}, have {len(last_sequence_df)}.")
        return

    cols_to_exclude = ['pdh', 'pdl', 'session_high', 'session_low', 'res_1_price', 'sup_1_price']
    cols_to_scale = [col for col in last_sequence_df.columns if col not in cols_to_exclude]
    sequence_scaled = last_sequence_df.copy()
    sequence_scaled[cols_to_scale] = scaler.transform(last_sequence_df[cols_to_scale])
    input_tensor = torch.tensor(sequence_scaled.values, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Prediction
    prediction_tensor = model(input_tensor)
        
    
    # 5. SHAP Explanation
    print("Calculating SHAP values...")
    background = torch.tensor(X_train[np.random.choice(X_train.shape[0], min(20, len(X_train)), replace=False)], dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(input_tensor)
    
    def explain_output(output_name, shap_values_for_output, effect_positive="HIGHER", effect_negative="LOWER"):
        shap_last_step = shap_values_for_output[0, -1, :]
        total_abs_shap = np.sum(np.abs(shap_last_step))
        top_indices = np.argsort(np.abs(shap_last_step))[::-1][:5]
        print(f"\n--- Top 5 Factors Influencing {output_name} ---")
        for i in top_indices:
            feature_name = feature_df.columns[i]
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
    print("--- Live Trading Assistant Prediction ---")
    print(f"Timestamp: {last_sequence_df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("\nPrediction:")
    print(f"  - Direction Bias:     {dir_pred:.2f} ({direction_text})")
    print(f"  - Predicted Quality:  {predicted_star} Stars")
    print(f"  - Predicted R:R:      {rr_pred:.2f}")
    print(f"  - SL Hit Probability: {sl_prob * 100:.2f}%")

    latest_data = last_sequence_df.iloc[-1]
    entry_price = latest_data['close']
    atr = latest_data['atr_14']
    level_names = {'pdh': "Previous Day's High", 'pdl': "Previous Day's Low", 'session_high': 'Recent Session High', 'session_low': 'Recent Session Low'}
    sl_reason_text = "N/A"
    
    if direction_text == "Buy":
        stop_loss_price = latest_data['sup_1_price'] - atr * 0.25
        risk_per_share = entry_price - stop_loss_price
        take_profit_price = entry_price + (risk_per_share * rr_pred)
        for level_col, name in level_names.items():
            if latest_data[level_col] == latest_data['sup_1_price']:
                sl_reason_text = f"Based on {name}"; break
    else: # Sell
        stop_loss_price = latest_data['res_1_price'] + atr * 0.25
        risk_per_share = stop_loss_price - entry_price
        take_profit_price = entry_price - (risk_per_share * rr_pred)
        for level_col, name in level_names.items():
            if latest_data[level_col] == latest_data['res_1_price']:
                sl_reason_text = f"Based on {name}"; break
    tp_reason_text = f"Calculated from Predicted {rr_pred:.2f} R:R"

    print("\nSuggested Trade Parameters:")
    print(f"  - Entry Price:        ~{entry_price:.5f}")
    print(f"  - Stop Loss:          ~{stop_loss_price:.5f} (Reason: {sl_reason_text})")
    print(f"  - Take Profit:        ~{take_profit_price:.5f} (Reason: {tp_reason_text})")
    
    explain_output("R:R", shap_values[:, :, :, 6], "HIGHER", "LOWER")
    explain_output("Direction", shap_values[:, :, :, 0], "more TOWARDS BUY", "more TOWARDS SELL")
    explain_output(f"Quality ({predicted_star} Stars)", shap_values[:, :, :, 1 + (predicted_star - 1)], "HIGHER", "LOWER")
    explain_output("SL Probability", shap_values[:, :, :, 7], "HIGHER", "LOWER")
        
    print("="*75 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GBP/JPY Trading Assistant")
    parser.add_argument('mode', choices=['train', 'infer'], help="The mode to run the script in: 'train' or 'infer'")
    args = parser.parse_args()
    if args.mode == 'train': do_training()
    elif args.mode == 'infer': do_inference()