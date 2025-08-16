import time
import subprocess
import re
from datetime import datetime
import pytz
from plyer import notification
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow informational messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from stable_baselines3 import PPO

# --- 1. Paper Trading Configuration ---
INITIAL_ACCOUNT_BALANCE = 100000.0
RISK_PER_TRADE_PERCENT = 0.01  # Risk 1% of the account per trade

# --- 2. Assistant Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_LOG_FILE = PROJECT_ROOT / "prediction_log.csv"
RL_MODEL_PATH = PROJECT_ROOT / "models/rl_trade_manager.zip"
TRADING_START_HOUR = 21; TRADING_END_HOUR = 1; TIMEZONE = 'America/Los_Angeles'
MIN_QUALITY = 4; MAX_SL_PROB = 0.30; MIN_RR = 2.5
SLEEP_INTERVAL_ACTIVE = 300; SLEEP_INTERVAL_INACTIVE = 900

# --- 3. Helper Functions (Self-Contained) ---

def load_and_preprocess_data(period_1h="2y", period_5m="59d"):
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
    df = df_5m.copy()
    df['atr_14'] = calculate_atr(df, period=14)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['returns'] = df['close'].pct_change()
    pst = pytz.timezone(TIMEZONE)
    df.index = df.index.tz_convert(pst)
    df['hour'] = df.index.hour
    df['is_trading_session'] = ((df['hour'] >= TRADING_START_HOUR) | (df['hour'] < TRADING_END_HOUR)).astype(int)
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

# --- 4. Paper Trade Manager Class ---
class PaperTradeManager:
    def __init__(self, balance, risk_percent):
        self.balance = balance
        self.risk_percent = risk_percent
        self.trade_open = False
        self.trade_info = {}
        self.usdjpy_rate = 150.0

    def update_usdjpy_rate(self):
        try:
            self.usdjpy_rate = yf.Ticker("USDJPY=X").history(period='1d')['Close'].iloc[-1]
        except Exception:
            print("  -> Warning: Could not fetch USD/JPY rate. Using last known rate.")

    def enter_trade(self, data, timestamp):
        if self.trade_open: return False
        self.update_usdjpy_rate()
        risk_amount_usd = self.balance * self.risk_percent
        risk_per_unit_jpy = abs(data['entry'] - data['sl'])
        if risk_per_unit_jpy == 0: return False
        risk_per_unit_usd = risk_per_unit_jpy / self.usdjpy_rate
        position_size = risk_amount_usd / risk_per_unit_usd

        self.trade_open = True
        self.trade_info = {
            "entry_time": timestamp, "entry_price": data['entry'],
            "sl_price": data['sl'], "tp_price": data['tp'],
            "position_size": position_size, "direction": 1 if "Buy" in data['direction_text'] else -1,
            "prediction_data": data, "open_pnl_usd": 0.0, "open_pnl_r": 0.0,
            "time_in_trade": 0
        }
        print("\n" + "*"*60); print(f"!!! PAPER TRADE ENTERED: {data['direction_text']} GBP/JPY !!!")
        print(f"  - Entry: {self.trade_info['entry_price']:.5f}, Size: {position_size:.2f} units")
        print(f"  - SL: {self.trade_info['sl_price']:.5f}, TP: {self.trade_info['tp_price']:.5f}")
        print(f"  - Risking ${risk_amount_usd:.2f} on this trade."); print("*"*60 + "\n")
        return True

    def update_pnl(self, current_price):
        if not self.trade_open: return
        pnl_jpy = (current_price - self.trade_info['entry_price']) * self.trade_info['direction'] * self.trade_info['position_size']
        self.trade_info['open_pnl_usd'] = pnl_jpy / self.usdjpy_rate
        risk_per_unit_jpy = abs(self.trade_info['entry_price'] - self.trade_info['sl_price'])
        self.trade_info['open_pnl_r'] = (pnl_jpy / risk_per_unit_jpy) if risk_per_unit_jpy > 0 else 0
        self.trade_info['time_in_trade'] += 1

    def close_trade(self, exit_price, timestamp, reason):
        if not self.trade_open: return
        self.update_pnl(exit_price)
        final_pnl_usd, final_outcome_rr = self.trade_info['open_pnl_usd'], self.trade_info['open_pnl_r']
        self.balance += final_pnl_usd
        self.trade_open = False
        print("\n" + "*"*60); print(f"!!! PAPER TRADE CLOSED !!!")
        print(f"  - Reason: {reason}"); print(f"  - Exit Price: {exit_price:.5f}")
        print(f"  - P/L: ${final_pnl_usd:+.2f} | Outcome: {final_outcome_rr:+.2f}R")
        print(f"  - New Account Balance: ${self.balance:.2f}"); print("*"*60 + "\n")
        self._log_trade(reason, final_outcome_rr)
        self.trade_info = {}

    def _log_trade(self, reason, outcome_rr):
        data = self.trade_info['prediction_data']
        log_entry = {
            'Timestamp': self.trade_info['entry_time'], 'Direction': data['direction_text'],
            'PredictedQuality': data['quality'], 'PredictedRR': data['rr'],
            'SLHitProbability': data['sl_prob'], 'EntryPrice': self.trade_info['entry_price'],
            'StopLossPrice': self.trade_info['sl_price'], 'TakeProfitPrice': data['tp'],
            'SLReason': data['sl_reason'], 'ActualOutcomeRR': f"{outcome_rr:.2f}",
            'Notes': f"Closed by {reason}"
        }
        log_df = pd.DataFrame([log_entry])
        if not PREDICTION_LOG_FILE.exists(): log_df.to_csv(PREDICTION_LOG_FILE, index=False)
        else: log_df.to_csv(PREDICTION_LOG_FILE, mode='a', header=False, index=False)
        print(f"  -> Trade outcome logged to {PREDICTION_LOG_FILE.name}")

# --- 2. The Live Assistant Engine ---

def parse_inference_output(output_text):
    # (This function is the same as before)
    try:
        direction_text = re.search(r"Direction Bias:.*?\((.*?)\)", output_text).group(1)
        quality = int(re.search(r"Predicted Quality:\s*(\d+)", output_text).group(1))
        rr = float(re.search(r"Predicted R:R:\s*([\d\.]+)", output_text).group(1))
        sl_prob = float(re.search(r"SL Hit Probability:\s*([\d\.]+)%", output_text).group(1)) / 100.0
        entry = float(re.search(r"Entry Price:\s*~\s*([\d\.]+)", output_text).group(1))
        sl = float(re.search(r"Stop Loss:\s*~\s*([\d\.]+)", output_text).group(1))
        tp = float(re.search(r"Take Profit:\s*~\s*([\d\.]+)", output_text).group(1))
        sl_reason = re.search(r"Stop Loss:.*?\((.*?)\)", output_text).group(1)
        return {
            "direction_text": direction_text, "quality": quality, "rr": rr, "sl_prob": sl_prob,
            "entry": entry, "sl": sl, "tp": tp, "sl_reason": sl_reason
        }
    except (AttributeError, IndexError, ValueError):
        return None

def format_alert_message(data):
    # (This function is the same as before)
    title = f"{data['quality']}-Star GBP/JPY {data['direction_text']} Setup Detected!"
    message = (
        f"Predicted R:R: {data['rr']:.2f}\n"
        f"SL Hit Probability: {data['sl_prob']:.1%}\n"
        f"-------------------------------------\n"
        f"Entry:  ~{data['entry']:.5f}\n"
        f"SL:     ~{data['sl']:.5f} ({data['sl_reason']})\n"
        f"TP:     ~{data['tp']:.5f}"
    )
    return title, message

def send_desktop_notification(title, message):
    # (This function is the same as before)
    try:
        notification.notify(title=title, message=message, app_name='GBP/JPY Trading Assistant', timeout=20)
        print("  -> Notification Sent!")
    except Exception as e:
        print(f"  -> Error sending notification: {e}")

# ** NEW FUNCTION TO LOG PREDICTIONS **
def log_prediction(data, timestamp):
    """Saves the details of an alert to the prediction_log.csv file."""
    log_entry = {
        'Timestamp': timestamp,
        'Direction': data['direction_text'],
        'PredictedQuality': data['quality'],
        'PredictedRR': data['rr'],
        'SLHitProbability': data['sl_prob'],
        'EntryPrice': data['entry'],
        'StopLossPrice': data['sl'],
        'TakeProfitPrice': data['tp'],
        'SLReason': data['sl_reason'],
        'ActualOutcomeRR': '', # For you to fill in later
        'Notes': ''              # For you to fill in later
    }
    
    log_df = pd.DataFrame([log_entry])
    
    # Append to the CSV file, create it with a header if it doesn't exist
    if not PREDICTION_LOG_FILE.exists():
        log_df.to_csv(PREDICTION_LOG_FILE, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_FILE, mode='a', header=False, index=False)
    
    print(f"  -> Prediction logged to {PREDICTION_LOG_FILE.name}")


def run_live_assistant():
    print("--- GBP/JPY Live Paper Trading Assistant ---")
    print(f"Initial Balance: ${INITIAL_ACCOUNT_BALANCE:.2f}, Risk: {RISK_PER_TRADE_PERCENT:.1%}")
    paper_trader = PaperTradeManager(INITIAL_ACCOUNT_BALANCE, RISK_PER_TRADE_PERCENT)
    try:
        rl_model = PPO.load(RL_MODEL_PATH)
        print("RL Trade Manager loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: RL model not found at {RL_MODEL_PATH}. Run 'src/run_rl_training.py' first.")
        return

    print("Assistant activated... (Press Ctrl+C to stop)")
    script_path = os.path.join('src', 'run_assistant.py')

    while True:
        try:
            tz = pytz.timezone(TIMEZONE)
            current_time = datetime.now(tz)
            is_trading_session = (current_time.hour >= TRADING_START_HOUR) or (current_time.hour < TRADING_END_HOUR)

            if is_trading_session:
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Active Session: Running check...")
                
                if paper_trader.trade_open:
                    df_1h, df_5m = load_and_preprocess_data(period_1h="5d", period_5m="3d")
                    if df_5m is not None:
                        feature_df = add_key_level_features(df_5m, df_1h)
                        latest_candle = feature_df.iloc[-1]
                        current_price = latest_candle['close']
                        info = paper_trader.trade_info
                        
                        paper_trader.update_pnl(current_price)
                        print(f"  -> Managing open trade. Current P/L: {paper_trader.trade_info['open_pnl_r']:+.2f}R")
                        
                        # Check for SL/TP breach first
                        if (info['direction'] == 1 and latest_candle['low'] <= info['sl_price']):
                            paper_trader.close_trade(info['sl_price'], current_time, "Stop Loss Hit")
                        elif (info['direction'] == 1 and latest_candle['high'] >= info['tp_price']):
                            paper_trader.close_trade(info['tp_price'], current_time, "Take Profit Hit")
                        elif (info['direction'] == -1 and latest_candle['high'] >= info['sl_price']):
                            paper_trader.close_trade(info['sl_price'], current_time, "Stop Loss Hit")
                        elif (info['direction'] == -1 and latest_candle['low'] <= info['tp_price']):
                            paper_trader.close_trade(info['tp_price'], current_time, "Take Profit Hit")
                        else:
                            # Build the state for the RL Agent
                            market_features = latest_candle.values
                            trade_features = np.array([
                                1.0, # Position is open
                                paper_trader.trade_info['time_in_trade'] / 288.0, # Max steps from env
                                paper_trader.trade_info['open_pnl_r']
                            ])
                            state = np.concatenate([market_features, trade_features]).astype(np.float32)
                            
                            # Get RL agent's decision
                            action, _ = rl_model.predict(state, deterministic=True)
                            if action == 1: # CLOSE action
                                paper_trader.close_trade(current_price, current_time, "RL Agent Exit")
                else: # If no trade is open, look for one
                    python_executable = sys.executable
                    process = subprocess.run(
                        [python_executable, script_path, 'infer'],
                        capture_output=True, text=True, check=False, encoding='utf-8'
                    )
                    if process.returncode == 0:
                        parsed_data = parse_inference_output(process.stdout) # You will need to add your parse_inference_output function
                        if parsed_data:
                            print(f"  -> LSTM Analysis Complete: Quality={parsed_data['quality']}*, R:R={parsed_data['rr']:.2f}")
                            is_high_quality_setup = (
                                parsed_data['quality'] >= MIN_QUALITY and
                                parsed_data['rr'] >= MIN_RR and
                                parsed_data['sl_prob'] <= MAX_SL_PROB
                            )
                            if is_high_quality_setup:
                                # (You will need to add your send_desktop_notification function)
                                paper_trader.enter_trade(parsed_data, current_time)
                
                time.sleep(SLEEP_INTERVAL_ACTIVE)
            else:
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Inactive Session: Sleeping...")
                if paper_trader.trade_open:
                    print(f"  -> WARNING: Trade open outside session. P/L: {paper_trader.trade_info['open_pnl_r']:+.2f}R")
                time.sleep(SLEEP_INTERVAL_INACTIVE)
        except KeyboardInterrupt:
            print("\nAssistant stopped by user. Exiting."); break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}"); time.sleep(60)

# You will need to copy your existing parse_inference_output, format_alert_message, 
# and send_desktop_notification functions into this file for it to be fully complete.

if __name__ == '__main__':
    # Due to the complexity, the main logic is now inside the run_live_assistant()
    # To use this script, you must first copy your existing helper functions
    # (parse_inference_output, etc.) into the space between the PaperTradeManager class
    # and the run_live_assistant function.
    run_live_assistant()