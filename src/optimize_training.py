import subprocess
import re
import sys
from pathlib import Path
import shutil

def run_pro_optimizer():
    """
    Runs the main training script multiple times to find the best model.
    """
    # Prompt for number of runs
    while True:
        try:
            num_runs = int(input("➡️ Enter number of runs (e.g., 5, 10): "))
            if num_runs > 0:
                break
            print("❌ Enter a number > 0.")
        except ValueError:
            print("❌ Enter a whole number.")
    
    print(f"🚀 Optimizing with {num_runs} runs...")
    
    # Setup paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'src' else SCRIPT_DIR
    assistant_script_path = PROJECT_ROOT / 'src' / 'run_assistant.py'
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data"

    # Find next version
    final_version = 1
    while (models_dir / f"gbpjpy_assistant_v{final_version}.pth").exists():
        final_version += 1
    print(f"✅ Saving champion as VERSION {final_version}.")

    # Setup temporary directory
    temp_dir = PROJECT_ROOT / "temp_optimizer"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "temp_model.pth"
    temp_scaler_path = temp_dir / "temp_scaler.pkl"
    temp_data_dir = temp_dir / "temp_processed_data"

    best_loss_overall = float('inf')
    champion_run_number = 0

    # Main optimization loop
    for i in range(num_runs):
        run_number = i + 1
        print(f"Run {run_number}/{num_runs}...")

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
            print(f"  -> Failed: Check temp_optimizer/error_run_{run_number}.log")
            with open(temp_dir / f"error_run_{run_number}.log", 'w', encoding='utf-8') as f:
                f.write(process.stderr)
            continue

        if not temp_model_path.exists():
            print(f"  -> Failed: No model file produced.")
            continue

        try:
            loss_matches = re.findall(r"New best model saved \(Val Loss: ([\d\.]+)\)", process.stdout)
            if not loss_matches:
                print(f"  -> Failed: No best model saved.")
                continue
            
            final_best_loss = float(loss_matches[-1])
            print(f"  -> Loss: {final_best_loss:.4f}")
            if final_best_loss < best_loss_overall:
                best_loss_overall = final_best_loss
                champion_run_number = run_number
                shutil.copy(temp_model_path, temp_dir / "champion.pth")
                shutil.copy(temp_scaler_path, temp_dir / "champion.pkl")
                if temp_data_dir.exists():
                    shutil.rmtree(temp_dir / "champion_data", ignore_errors=True)
                    shutil.copytree(temp_data_dir, temp_dir / "champion_data")
        except (IndexError, ValueError) as e:
            print(f"  -> Failed: Parsing error: {e}")
            continue

    # Finalize and clean up
    print("\n🏁 Optimization Complete!")
    champion_model_temp_path = temp_dir / "champion.pth"
    if champion_model_temp_path.exists():
        final_model_path = models_dir / f"gbpjpy_assistant_v{final_version}.pth"
        final_scaler_path = models_dir / f"scaler_v{final_version}.pkl"
        final_data_dir = data_dir / f"processed_for_training_v{final_version}"
        
        print(f"🏆 Best Run: {champion_run_number} (Loss: {best_loss_overall:.4f})")
        print(f"   Model: {final_model_path.name}")
        print(f"   Scaler: {final_scaler_path.name}")
        print(f"   Data: {final_data_dir.name}")

        champion_model_temp_path.rename(final_model_path)
        (temp_dir / "champion.pkl").rename(final_scaler_path)
        (temp_dir / "champion_data").rename(final_data_dir)
    else:
        print("No successful runs completed.")

    print("Cleaning up...")
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    run_pro_optimizer()