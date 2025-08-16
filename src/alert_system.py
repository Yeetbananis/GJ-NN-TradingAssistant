import time
import subprocess
import re
from datetime import datetime
import pytz
from plyer import notification

# --- 1. Configuration: Adjust these rules to match your trading plan ---
TRADING_START_HOUR = 21
TRADING_END_HOUR = 1
TIMEZONE = 'America/Los_Angeles'
MIN_QUALITY = 4
MAX_SL_PROB = 0.30
MIN_RR = 2.0
SLEEP_INTERVAL_ACTIVE = 300
SLEEP_INTERVAL_INACTIVE = 600

# --- 2. The Alert Engine ---

def parse_inference_output(output_text):
    """
    Parses the text output from the inference script to extract key values.
    Returns a dictionary of the parsed values or None if parsing fails.
    """
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
            "direction_text": direction_text,
            "quality": quality,
            "rr": rr,
            "sl_prob": sl_prob,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_reason": sl_reason
        }
    except (AttributeError, IndexError, ValueError) as e:
        print(f"  -> Warning: Could not parse inference output. Error: {e}")
        return None

def run_alert_engine():
    """
    The main loop that runs the assistant, checks for setups, and sends alerts.
    """
    print("--- GBP/JPY Trading Assistant ---")
    print(f"Alert Rules: Quality >= {MIN_QUALITY} Stars | R:R >= {MIN_RR} | SL Prob <= {MAX_SL_PROB*100}%")
    print("Assistant activated. Monitoring for trading session...")
    
    alert_sent_for_current_setup = False

    while True:
        try:
            tz = pytz.timezone(TIMEZONE)
            current_time = datetime.now(tz)
            current_hour = current_time.hour
            
            is_trading_session = (current_hour >= TRADING_START_HOUR) or (current_hour < TRADING_END_HOUR)

            if is_trading_session:
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Active Session: Running check...")
                
                process = subprocess.run(
                    ['python', 'src/run_assistant.py', 'infer'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if process.returncode != 0:
                    print("  -> Error: Inference script failed to run.")
                    print(process.stderr)
                else:
                    inference_output = process.stdout
                    parsed_data = parse_inference_output(inference_output)
                    
                    if parsed_data:
                        print(f"  -> Analysis Complete: Quality={parsed_data['quality']}*, R:R={parsed_data['rr']:.2f}, SL%={parsed_data['sl_prob']:.2%}")
                        
                        is_high_quality_setup = (
                            parsed_data['quality'] >= MIN_QUALITY and
                            parsed_data['rr'] >= MIN_RR and
                            parsed_data['sl_prob'] <= MAX_SL_PROB
                        )
                        
                        if is_high_quality_setup and not alert_sent_for_current_setup:
                            print("\n" + "!"*60)
                            print("!!! HIGH-QUALITY TRADING SETUP DETECTED !!!")
                            print("!"*60)
                            print(inference_output)
                            alert_sent_for_current_setup = True
                        
                        elif not is_high_quality_setup:
                            if alert_sent_for_current_setup:
                                print("  -> Previous setup is no longer valid. Resetting alert status.")
                            alert_sent_for_current_setup = False
                
                print(f"  -> Sleeping for {SLEEP_INTERVAL_ACTIVE // 60} minutes...")
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
    run_alert_engine()