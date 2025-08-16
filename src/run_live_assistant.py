import time
import subprocess
import re
from datetime import datetime
import pytz
from plyer import notification
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow informational messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from stable_baselines3 import PPO
from run_rl_training import TradingEnv
from broker_connector import BrokerConnector
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pendulum
import requests
import configparser
import logging
import queue

# --- 1. Paper Trading Configuration ---
RISK_PER_TRADE_PERCENT = 0.01  # Risk 1% of the account per trade

# --- 2. Assistant Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_LOG_FILE = PROJECT_ROOT / "prediction_log.csv"
RL_MODEL_PATH = PROJECT_ROOT / "models/rl_trade_manager.zip"
TRADING_START_HOUR = 21  # 1 PM PDT for testing is 13
TRADING_END_HOUR = 1   # 3 PM PDT for testing is 15
TIMEZONE = 'America/Los_Angeles'
MIN_QUALITY = 4
MAX_SL_PROB = 0.30
MIN_RR = 2.5
SLEEP_INTERVAL_ACTIVE = 300
SLEEP_INTERVAL_INACTIVE = 900
DATA_CACHE_1H = PROJECT_ROOT / "data/cache/GBPJPY=X_1h_cache.csv"
DATA_CACHE_5M = PROJECT_ROOT / "data/cache/GBPJPY=X_5m_cache.csv"

# --- 3. Telegram Setup ---
class PrintLogger:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.message_queue = queue.Queue()
        self.last_sent = time.time()

    def escape_markdown(self, text):
        """Escape special Markdown characters to prevent BadRequest errors."""
        escape_chars = ['*', '_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in escape_chars:
            text = text.replace(char, f'\\{char}')
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logging.warning(f"Retrying Telegram send (attempt {retry_state.attempt_number})...")
    )
    def send_message(self, text):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"```{self.escape_markdown(text)}```",
            "parse_mode": "MarkdownV2"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def write(self, message):
        sys.__stdout__.write(message)  # Print to terminal
        if message.strip():
            self.message_queue.put(message.strip())
            if time.time() - self.last_sent >= 5:  # Send every 5 seconds
                messages = []
                while not self.message_queue.empty():
                    messages.append(self.message_queue.get())
                if messages:
                    message_to_send = "\n".join(messages)
                    try:
                        self.send_message(message_to_send)
                        self.last_sent = time.time()
                    except Exception as e:
                        logging.error(f"Failed to send Telegram message: {e}")

    def flush(self):
        sys.__stdout__.flush()
        if time.time() - self.last_sent >= 5 and not self.message_queue.empty():
            messages = []
            while not self.message_queue.empty():
                messages.append(self.message_queue.get())
            message_to_send = "\n".join(messages)
            try:
                self.send_message(message_to_send)
                self.last_sent = time.time()
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

# Initialize Telegram logger
try:
    config_path = PROJECT_ROOT / 'src' / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    TELEGRAM_BOT_TOKEN = config['telegram']['bot_token']
    TELEGRAM_CHAT_ID = config['telegram']['chat_id']
    sys.stdout = PrintLogger(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
except Exception as e:
    print(f"WARNING: Could not initialize Telegram logger: {e}. Using console output only.")

# --- 4. Helper Functions ---
def validate_timezone():
    """Validates local time against an external time source."""
    try:
        response = requests.get("http://worldtimeapi.org/api/timezone/America/Los_Angeles", timeout=5)
        response.raise_for_status()
        api_time = pendulum.parse(response.json()['datetime'])
        local_time = pendulum.now('America/Los_Angeles')
        time_diff = abs((local_time - api_time).total_seconds())
        if time_diff > 60:
            print(f"  -> Warning: Local time differs from API time by {time_diff:.0f} seconds. Using API time.")
            return api_time
        return local_time
    except Exception as e:
        print(f"  -> Warning: Could not validate time with API: {e}. Using local time.")
        return pendulum.now('America/Los_Angeles')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_yf_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError("Empty data returned from yfinance")
    return df

def load_and_preprocess_data(period_1h="2y", period_5m="59d"):
    print("Loading/Downloading raw data...")
    raw_data_dir = PROJECT_ROOT / "data/raw"
    cache_data_dir = PROJECT_ROOT / "data/cache"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    cache_data_dir.mkdir(parents=True, exist_ok=True)
    path_1h = raw_data_dir / "GBPJPY=X_1h.csv"
    path_5m = raw_data_dir / "GBPJPY=X_5m.csv"
    cache_1h = DATA_CACHE_1H
    cache_5m = DATA_CACHE_5M

    def load_from_cache(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception:
            return None

    try:
        df_1h = download_yf_data("GBPJPY=X", period_1h, "1h")
        df_1h.to_csv(path_1h)
        df_1h.to_csv(cache_1h)
        df_5m = download_yf_data("GBPJPY=X", period_5m, "5m")
        df_5m.to_csv(path_5m)
        df_5m.to_csv(cache_5m)
    except Exception as e:
        print(f"  -> Warning: Failed to download fresh data: {e}. Attempting to load from cache...")
        df_1h = load_from_cache(cache_1h)
        df_5m = load_from_cache(cache_5m)
        if df_1h is None or df_5m is None:
            print("  -> ERROR: No cached data available or cache corrupted.")
            return None, None

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
    pst = pendulum.timezone(TIMEZONE)
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
        if res_level:
            res_prices[i] = min(res_level)
        if sup_level:
            sup_prices[i] = max(sup_level)
    df['res_1_price'] = res_prices
    df['sup_1_price'] = sup_prices
    df['dist_to_res_1'] = (df['res_1_price'] / df['close']) - 1
    df['dist_to_sup_1'] = (df['sup_1_price'] / df['close']) - 1
    return df.dropna()

def parse_inference_output(output_text):
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
    title = f"🚨 {data['quality']}-Star GBP/JPY {data['direction_text']} Setup Detected! 🚨"
    message = (
        f"*Predicted R:R*: {data['rr']:.2f}\n"
        f"*SL Hit Probability*: {data['sl_prob']:.1%}\n"
        f"-------------------------------------\n"
        f"*Entry*: ~{data['entry']:.5f}\n"
        f"*Stop Loss*: ~{data['sl']:.5f} ({data['sl_reason']})\n"
        f"*Take Profit*: ~{data['tp']:.5f}"
    )
    return title, message

@retry(
         stop=stop_after_attempt(5),  # Increased retries
         wait=wait_exponential(multiplier=1, min=2, max=15),
         retry=retry_if_exception_type(Exception),
         before_sleep=lambda retry_state: print(f"Retrying Telegram notification send (attempt {retry_state.attempt_number})...")
     )
def send_telegram_notification(title, message):
         url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
         # Simplified escaping to avoid MarkdownV2 issues
         escape_chars = ['*', '_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
         escaped_title = title
         escaped_message = message
         for char in escape_chars:
             escaped_title = escaped_title.replace(char, f'\\{char}')
             escaped_message = escaped_message.replace(char, f'\\{char}')
         payload = {
             "chat_id": TELEGRAM_CHAT_ID,
             "text": f"*{escaped_title}*\n\n{escaped_message}",
             "parse_mode": "MarkdownV2",
             "disable_notification": False
         }
         try:
             response = requests.post(url, json=payload, timeout=15)  # Increased timeout
             response.raise_for_status()
             print(f"  -> Telegram response: {response.json()}")
             return response.json()
         except requests.exceptions.RequestException as e:
             print(f"  -> Detailed Telegram error: {type(e).__name__}: {str(e)}")
             raise

def send_desktop_notification(title, message):
    try:
        notification.notify(title=title, message=message, app_name='GBP/JPY Trading Assistant', timeout=20)
        print("  -> Desktop Notification Sent!")
    except Exception as e:
        print(f"  -> Error sending desktop notification: {e}")
    try:
        send_telegram_notification(title, message)
        print("  -> Telegram Notification Sent!")
    except Exception as e:
        print(f"  -> Error sending Telegram notification: {e}")

def log_prediction(data, timestamp, trade_id=None, update=False, outcome_rr=None, notes=None):
    log_entry = {
        'Timestamp': timestamp,
        'Direction': data['direction_text'] if not update else data.get('direction_text', ''),
        'PredictedQuality': data['quality'] if not update else data.get('quality', ''),
        'PredictedRR': data['rr'] if not update else data.get('rr', ''),
        'SLHitProbability': data['sl_prob'] if not update else data.get('sl_prob', ''),
        'EntryPrice': data['entry'] if not update else data.get('entry', ''),
        'StopLossPrice': data['sl'] if not update else data.get('sl', ''),
        'TakeProfitPrice': data['tp'] if not update else data.get('tp', ''),
        'SLReason': data['sl_reason'] if not update else data.get('sl_reason', ''),
        'TradeID': trade_id if trade_id else '',
        'ActualOutcomeRR': outcome_rr if outcome_rr is not None else '',
        'Notes': notes if notes else ''
    }
    log_df = pd.DataFrame([log_entry])
    if update and PREDICTION_LOG_FILE.exists():
        existing_log = pd.read_csv(PREDICTION_LOG_FILE)
        if trade_id and 'TradeID' in existing_log.columns:
            existing_log.loc[existing_log['TradeID'] == trade_id, ['ActualOutcomeRR', 'Notes']] = [outcome_rr, notes]
            existing_log.to_csv(PREDICTION_LOG_FILE, index=False)
        else:
            log_df.to_csv(PREDICTION_LOG_FILE, mode='a', header=False, index=False)
    else:
        if not PREDICTION_LOG_FILE.exists():
            log_df.to_csv(PREDICTION_LOG_FILE, index=False)
        else:
            log_df.to_csv(PREDICTION_LOG_FILE, mode='a', header=False, index=False)
    print(f"  -> {'Updated' if update else 'Logged'} trade to {PREDICTION_LOG_FILE.name}")

def run_live_assistant():
    print("--- GBP/JPY Live Trading Assistant (OANDA Integrated) ---")
    print(f"Alert Rules: Quality >= {MIN_QUALITY} Stars | R:R >= {MIN_RR} | SL Prob <= {MAX_SL_PROB*100}%")
    
    broker = BrokerConnector()
    if not broker.client:
        print("Exiting due to broker connection failure.")
        return

    try:
        rl_model = PPO.load(RL_MODEL_PATH)
        print("RL Trade Manager loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: RL model not found at {RL_MODEL_PATH}. Run 'src/run_rl_training.py' first.")
        return

    print("Assistant activated. Monitoring for trading session... (Press Ctrl+C to stop)")
    script_path = os.path.join('src', 'run_assistant.py')

    while True:
        try:
            current_time = validate_timezone()
            is_trading_session = (current_time.hour >= TRADING_START_HOUR) or (current_time.hour < TRADING_END_HOUR)

            if is_trading_session:
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Active Session: Running check...")
                
                open_trade = broker.get_open_trade("GBP_JPY")

                if open_trade:
                    print("  -> Managing open trade with RL Agent...")
                    df_1h, df_5m = load_and_preprocess_data(period_1h="5d", period_5m="3d")
                    if df_1h is None or df_5m is None:
                        print("  -> ERROR: Could not load market data for RL state. Skipping trade management.")
                        time.sleep(SLEEP_INTERVAL_ACTIVE)
                        continue

                    feature_df = add_key_level_features(df_5m, df_1h)
                    if feature_df.empty:
                        print("  -> ERROR: No valid features for RL state. Skipping trade management.")
                        time.sleep(SLEEP_INTERVAL_ACTIVE)
                        continue

                    current_price = float(open_trade['price'])
                    entry_price = float(open_trade['initialUnits']) / abs(float(open_trade['initialUnits'])) * float(open_trade['price'])
                    trade_direction = 1 if float(open_trade['initialUnits']) > 0 else -1
                    live_pnl = float(open_trade['unrealizedPL'])
                    latest_data = feature_df.iloc[-1]
                    atr = latest_data['atr_14']
                    risk_per_share = abs(entry_price - float(open_trade['stopLoss']['price'])) if open_trade.get('stopLoss') else atr
                    if not open_trade.get('stopLoss'):
                        print(f"  -> Warning: No stopLoss in trade data. Using ATR ({atr:.5f}) for risk calculation.")
                    market_features = feature_df.iloc[-1].values.astype(np.float32)
                    position_status = 1.0
                    steps_in_trade = min(int((current_time - pd.to_datetime(open_trade['openTime']).tz_convert(TIMEZONE)).total_seconds() / 300), 288)
                    time_in_trade_norm = steps_in_trade / 288
                    pnl_in_r = (live_pnl / (abs(risk_per_share) * abs(float(open_trade['initialUnits'])))) * trade_direction if risk_per_share > 0 else 0.0
                    state = np.concatenate([market_features, [position_status, time_in_trade_norm, pnl_in_r]]).astype(np.float32)
                    action, _ = rl_model.predict(state, deterministic=True)
                    print(f"  -> Current P/L: ${live_pnl:+.2f} | RL Action: {'Close' if action == 1 else 'Hold'}")

                    if action == 1:
                        trade_id = open_trade['id']
                        response = broker.close_trade(trade_id)
                        if response:
                            close_price = float(response.get('price', current_price))
                            outcome_rr = (close_price - entry_price) / risk_per_share * trade_direction if risk_per_share > 0 else 0.0
                            notes = "Closed by RL Agent"
                            log_prediction(
                                data={},  # Empty dict since parsed_data may not be available
                                timestamp=validate_timezone(),
                                trade_id=trade_id,
                                update=True,
                                outcome_rr=outcome_rr,
                                notes=notes
                            )
                            print(f"  -> Trade {trade_id} closed by RL Agent. Outcome R:R: {outcome_rr:.2f}")
                        else:
                            print(f"  -> ERROR: Failed to close trade {trade_id}.")

                else:
                    python_executable = sys.executable
                    process = subprocess.run(
                        [python_executable, script_path, 'infer'],
                        capture_output=True, text=True, check=False, encoding='utf-8'
                    )
                    
                    if process.returncode == 0:
                        parsed_data = parse_inference_output(process.stdout)
                        if parsed_data:
                            print(f"  -> LSTM Analysis Complete: Quality={parsed_data['quality']}*, R:R={parsed_data['rr']:.2f}")
                            is_high_quality_setup = (
                                parsed_data['quality'] >= MIN_QUALITY and
                                parsed_data['rr'] >= MIN_RR and
                                parsed_data['sl_prob'] <= MAX_SL_PROB
                            )
                            if is_high_quality_setup:
                                title, message = format_alert_message(parsed_data)
                                send_desktop_notification(title, message)
                                
                                print("  -> High-quality setup found. Preparing to send order to OANDA...")
                                current_balance = broker.get_account_balance()
                                if current_balance:
                                    try:
                                        usdjpy_rate = yf.Ticker("USDJPY=X").history(period='1d')['Close'].iloc[-1]
                                    except Exception:
                                        print("  -> Warning: yfinance failed to fetch USD/JPY rate. Attempting OANDA API fallback...")
                                        price_data = broker.get_current_price("USD_JPY")
                                        if price_data and 'bid' in price_data and 'ask' in price_data:
                                            usdjpy_rate = (float(price_data['bid']) + float(price_data['ask'])) / 2
                                            print("  -> Fallback successful: USD/JPY rate from OANDA.")
                                        else:
                                            print("  -> ERROR: Could not fetch USD/JPY rate from OANDA. Aborting trade.")
                                            continue
                                    risk_amount_usd = current_balance * RISK_PER_TRADE_PERCENT
                                    risk_per_unit_jpy = abs(parsed_data['entry'] - parsed_data['sl'])
                                    if risk_per_unit_jpy > 0:
                                        risk_per_unit_usd = risk_per_unit_jpy / usdjpy_rate
                                        position_size = int(risk_amount_usd / risk_per_unit_usd)
                                        direction = 1 if "Buy" in parsed_data['direction_text'] else -1
                                        units = position_size * direction
                                        print(f"  -> Account Balance: ${current_balance:,.2f}")
                                        print(f"  -> Risk Amount: ${risk_amount_usd:,.2f} ({RISK_PER_TRADE_PERCENT:.0%})")
                                        print(f"  -> Calculated Position Size: {position_size} units")
                                        response = broker.create_market_order(
                                            instrument="GBP_JPY",
                                            units=units,
                                            sl_price=parsed_data['sl'],
                                            tp_price=parsed_data['tp']
                                        )
                                        trade_id = response.get('orderFillTransaction', {}).get('tradeOpened', {}).get('tradeID') if response else None
                                        log_prediction(parsed_data, current_time, trade_id=trade_id)
                                    else:
                                        print("  -> ERROR: Risk per unit is zero. Aborting trade.")
                                else:
                                    print("  -> ERROR: Could not fetch account balance. Aborting trade.")
                    else:
                        print("  -> Error: Inference script failed.")
                        print(process.stderr)
                
                time.sleep(SLEEP_INTERVAL_ACTIVE)

            else:
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Inactive Session: Sleeping...")
                time.sleep(SLEEP_INTERVAL_INACTIVE)

        except KeyboardInterrupt:
            print("\nAssistant stopped by user. Exiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_live_assistant()