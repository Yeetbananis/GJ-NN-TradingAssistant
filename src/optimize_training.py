import subprocess
import re
import sys
from pathlib import Path
import shutil

# --- Configuration ---
# NUM_RUNS is now set by user input
# ---

def run_pro_optimizer():
    """
    Runs the main training script multiple times...
    """
    # --- NEW: Prompt for number of runs ---
    while True:
        try:
            # Prompt the user for input
            user_input = input("➡️  Enter the number of optimization runs to perform (e.g., 5, 10): ")
            num_runs = int(user_input)
            if num_runs > 0:
                break # Exit the loop if input is a valid positive number
            else:
                print("❌ Please enter a number greater than zero.")
        except ValueError:
            # This runs if the user enters text instead of a number
            print("❌ Invalid input. Please enter a whole number.")
    # --- END: Prompt ---
    print("="*60)
    print(f"🚀 Starting PRO optimization run for {num_runs} training sessions...")
    print("="*60)
    
    # --- 1. ROBUST PATHING & VERSIONING ---
    # Correctly find the project root, even if this script is in the 'src' folder.
    SCRIPT_DIR = Path(__file__).resolve().parent
    if SCRIPT_DIR.name == 'src':
        PROJECT_ROOT = SCRIPT_DIR.parent
    else:
        PROJECT_ROOT = SCRIPT_DIR
    
    assistant_script_path = PROJECT_ROOT / 'src' / 'run_assistant.py'
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data"

    # Determine the next available version number BEFORE training starts.
    # This fixes the bug where it always defaulted to v1.
    final_version = 1
    while True:
        if not (models_dir / f"gbpjpy_assistant_v{final_version}.pth").exists():
            break
        final_version += 1
    print(f"✅ Next available model slot is VERSION {final_version}. The champion will be saved here.")

    # --- 2. SETUP TEMPORARY DIRECTORY ---
    temp_dir = PROJECT_ROOT / "temp_optimizer"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "temp_model.pth"
    temp_scaler_path = temp_dir / "temp_scaler.pkl"
    temp_data_dir = temp_dir / "temp_processed_data"

    best_loss_overall = float('inf')
    champion_run_number = 0

    # --- 3. MAIN OPTIMIZATION LOOP ---
    for i in range(num_runs):
        run_number = i + 1
        print(f"\n--- Starting Training Run {run_number}/{num_runs} ---")

        process = subprocess.run(
            [
                sys.executable, str(assistant_script_path), 'train',
                '--model-path', str(temp_model_path),
                '--scaler-path', str(temp_scaler_path),
                '--data-dir', str(temp_data_dir)
            ],
            capture_output=True, text=True, encoding='utf-8'
        )

        if process.returncode != 0:
            print(f"❌ Run {run_number} failed. Error log from subprocess:\n{process.stderr}")
            continue
            
        if not temp_model_path.exists():
            print(f"⚠️ Run {run_number} completed but did not produce a model file. Skipping.")
            continue

        try:
            loss_matches = re.findall(r"New best model saved \(Val Loss: ([\d\.]+)\)", process.stdout)
            if not loss_matches:
                print(f"⚠️ Run {run_number} finished but no best model was saved. Skipping.")
                continue
            
            final_best_loss = float(loss_matches[-1])
            print(f"--- [OK] Run {run_number} Complete | Final Best Loss: {final_best_loss:.4f} ---")

            if final_best_loss < best_loss_overall:
                print(f"   - 🏆 NEW CHAMPION! This is the best model so far.")
                best_loss_overall = final_best_loss
                champion_run_number = run_number
                # Copy the temp files to a champion holding location inside the temp folder
                shutil.copy(temp_model_path, temp_dir / "champion.pth")
                shutil.copy(temp_scaler_path, temp_dir / "champion.pkl")
                if temp_data_dir.exists():
                    shutil.rmtree(temp_dir / "champion_data", ignore_errors=True)
                    shutil.copytree(temp_data_dir, temp_dir / "champion_data")

        except (IndexError, ValueError) as e:
            print(f"⚠️ Error parsing output for Run {run_number}: {e}")

    # --- 4. FINALIZE AND CLEAN UP ---
    print("\n" + "="*60)
    print("🏁 Optimization Run Finished!")
    
    champion_model_temp_path = temp_dir / "champion.pth"
    if champion_model_temp_path.exists():
        final_model_path = models_dir / f"gbpjpy_assistant_v{final_version}.pth"
        final_scaler_path = models_dir / f"scaler_v{final_version}.pkl"
        final_data_dir = data_dir / f"processed_for_training_v{final_version}"
        
        print(f"🏆 The best model was from Run {champion_run_number} with a Validation Loss of {best_loss_overall:.4f}")
        print(f"   Saving final artifacts as VERSION {final_version}")

        # Move the champion files to their final, versioned destination
        champion_model_temp_path.rename(final_model_path)
        (temp_dir / "champion.pkl").rename(final_scaler_path)
        (temp_dir / "champion_data").rename(final_data_dir)
        
        print(f"   - Final Model: {final_model_path.name}")
    else:
        print("No successful training runs were completed to produce a champion model.")

    print("   Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("="*60)


if __name__ == '__main__':
    run_pro_optimizer()