import time
import subprocess
import re
from datetime import datetime
import pytz
from plyer import notification
import sys
import os
import pandas as pd
from pathlib import Path

# --- 1. Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTION_LOG_FILE = PROJECT_ROOT / "prediction_log.csv"

TRADING_START_HOUR = 21
TRADING_END_HOUR = 1
TIMEZONE = 'America/Los_Angeles'
MIN_QUALITY = 4
MAX_SL_PROB = 0.30
MIN_RR = 2.5
SLEEP_INTERVAL_ACTIVE = 300
SLEEP_INTERVAL_INACTIVE = 900

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
    """The main loop that runs the assistant, checks for setups, and sends alerts."""
    print("--- GBP/JPY Live Trading Assistant ---")
    print(f"Alert Rules: Quality >= {MIN_QUALITY} Stars | R:R >= {MIN_RR} | SL Prob <= {MAX_SL_PROB*100}%")
    print("Assistant activated. Monitoring for trading session... (Press Ctrl+C to stop)")
    
    alert_sent_for_current_setup = False
    script_path = os.path.join('src', 'run_assistant.py')

    while True:
        try:
            tz = pytz.timezone(TIMEZONE)
            current_time = datetime.now(tz)
            current_hour = current_time.hour
            is_trading_session = (current_hour >= TRADING_START_HOUR) or (current_hour < TRADING_END_HOUR)

            if is_trading_session:
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Active Session: Running check...")
                python_executable = sys.executable
                process = subprocess.run(
                    [python_executable, script_path, 'infer'],
                    capture_output=True, text=True, check=False, encoding='utf-8'
                )
                
                if process.returncode != 0:
                    print("  -> Error: Inference script failed to run."); print(process.stderr)
                else:
                    inference_output = process.stdout
                    parsed_data = parse_inference_output(inference_output)
                    if parsed_data:
                        print(f"  -> Analysis Complete: Quality={parsed_data['quality']}*, R:R={parsed_data['rr']:.2f}, SL%={parsed_data['sl_prob']:.1%}")
                        is_high_quality_setup = (
                            parsed_data['quality'] >= MIN_QUALITY and
                            parsed_data['rr'] >= MIN_RR and
                            parsed_data['sl_prob'] <= MAX_SL_PROB
                        )
                        if is_high_quality_setup and not alert_sent_for_current_setup:
                            print("  -> !!! HIGH-QUALITY SETUP DETECTED !!!")
                            title, message = format_alert_message(parsed_data)
                            send_desktop_notification(title, message)
                            # ** LOG THE PREDICTION WHEN THE ALERT IS SENT **
                            log_prediction(parsed_data, current_time)
                            alert_sent_for_current_setup = True
                        elif not is_high_quality_setup:
                            if alert_sent_for_current_setup:
                                print("  -> Previous setup is no longer valid. Resetting alert status.")
                            alert_sent_for_current_setup = False
                time.sleep(SLEEP_INTERVAL_ACTIVE)

            else:
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Inactive Session: Sleeping...")
                alert_sent_for_current_setup = False
                time.sleep(SLEEP_INTERVAL_INACTIVE)

        except KeyboardInterrupt:
            print("\nAssistant stopped by user. Exiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Restarting check in 1 minute...")
            time.sleep(60)

if __name__ == '__main__':
    run_live_assistant()