# tune_with_cv.py
import subprocess
import re
import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import shutil
# Import the necessary functions directly from your script
# Note: Ensure this script is in your project's root or 'src' folder
try:
    from src.run_assistant import load_and_preprocess_data, add_key_level_features, SEQUENCE_LENGTH, LABEL_FILE
except ImportError:
    from run_assistant import load_and_preprocess_data, add_key_level_features, SEQUENCE_LENGTH, LABEL_FILE


def run_cv_tuning():
    """
    Performs a robust hyperparameter search using K-Fold Cross-Validation.
    This script ONLY identifies the best parameters; it does not save a final model.
    """
    # --- 1. DEFINE YOUR SEARCH SPACE ---
    param_grid = {
        'lr': [0.001, 0.0005],
        'batch_size': [16, 32],
        'hidden_size': [64, 128]
    }
    N_SPLITS = 5  # Number of folds for cross-validation
    # ---

    # --- 2. SETUP PATHS ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    if SCRIPT_DIR.name == 'src':
        PROJECT_ROOT = SCRIPT_DIR.parent
    else:
        PROJECT_ROOT = SCRIPT_DIR
    assistant_script_path = PROJECT_ROOT / 'src' / 'run_assistant.py'
    
    # Use a temporary directory for all intermediate files
    temp_dir = PROJECT_ROOT / "temp_cv_tuning"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "temp_model.pth"
    temp_scaler_path = temp_dir / "temp_scaler.pkl"
    temp_data_dir = temp_dir / "temp_processed_data"

    # --- 3. PREPARE THE FULL DATASET ONCE ---
    print("[OK] Preparing the full dataset for cross-validation...")
    df_1h, df_5m = load_and_preprocess_data(period_1h="730d", period_5m="59d")
    if df_1h is None: return
    feature_df = add_key_level_features(df_5m, df_1h)
    label_df = pd.read_csv(LABEL_FILE)
    
    valid_labels = label_df.dropna(subset=['entry_price']).copy()
    valid_labels['date_parsed'] = pd.to_datetime(valid_labels['date'])
    aligned_labels_list = []
    for _, label_row in valid_labels.iterrows():
        label_date = label_row['date_parsed'].date()
        entry_price = label_row['entry_price']
        session_features = feature_df[(feature_df.index.date == label_date)]
        if session_features.empty: continue
        closest_entry_time = (session_features['close'] - entry_price).abs().idxmin()
        new_label_entry = label_row.drop(['date', 'date_parsed']).to_dict()
        new_label_entry['timestamp'] = closest_entry_time
        aligned_labels_list.append(new_label_entry)
    aligned_labels_df = pd.DataFrame(aligned_labels_list).set_index('timestamp')

    X, y_dict = [], {col: [] for col in aligned_labels_df.columns}
    for timestamp, row in aligned_labels_df.iterrows():
        try:
            end_idx = feature_df.index.get_loc(timestamp)
            start_idx = end_idx - SEQUENCE_LENGTH + 1
            if start_idx < 0: continue
            X.append(feature_df.iloc[start_idx:end_idx + 1].values)
            for col, val in row.items(): y_dict[col].append(val)
        except KeyError: continue
    X = np.array(X)
    print(f"[OK] Dataset prepared with {len(X)} total samples.")
    
    # --- 4. RUN THE GRID SEARCH WITH CROSS-VALIDATION ---
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🚀 Starting Cross-Validation Grid Search for {len(hyperparameter_combinations)} combinations...")
    results = []

    for i, params in enumerate(hyperparameter_combinations):
        combo_number = i + 1
        print(f"\n--- Testing Combo {combo_number}/{len(hyperparameter_combinations)}: {params} ---")
        
        fold_losses = []
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_number = fold + 1
            print(f"  -> Running Fold {fold_number}/{N_SPLITS}...")
            
            train_indices_path = temp_dir / "train_idx.npy"
            val_indices_path = temp_dir / "val_idx.npy"
            np.save(train_indices_path, train_idx)
            np.save(val_indices_path, val_idx)

            command = [
                sys.executable, str(assistant_script_path), 'train',
                '--lr', str(params['lr']),
                '--batch-size', str(params['batch_size']),
                '--hidden-size', str(params['hidden_size']),
                '--train-indices-path', str(train_indices_path),
                '--val-indices-path', str(val_indices_path),
                # Use temp paths to force no versioning
                '--model-path', str(temp_model_path),
                '--scaler-path', str(temp_scaler_path),
                '--data-dir', str(temp_data_dir)
            ]
            
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

            if process.returncode == 0:
                loss_matches = re.findall(r"New best model saved \(Val Loss: ([\d\.]+)\)", process.stdout)
                if loss_matches:
                    best_fold_loss = float(loss_matches[-1])
                    fold_losses.append(best_fold_loss)
                    print(f"     Fold {fold_number} Best Loss: {best_fold_loss:.4f}")
                else:
                    fold_losses.append(float('inf'))
            else:
                print(f"     ❌ Fold {fold_number} failed. Check error log above.")
                fold_losses.append(float('inf'))

        if fold_losses:
            valid_losses = [loss for loss in fold_losses if loss != float('inf')]
            if valid_losses:
                avg_loss = np.mean(valid_losses)
                results.append({'params': params, 'avg_loss': avg_loss})
                print(f"  -> Combo {combo_number} Average Loss across {len(valid_losses)} folds: {avg_loss:.4f}")

    # --- 5. DISPLAY FINAL RESULTS ---
    print("\n" + "="*70)
    print("🏁 Cross-Validation Hyperparameter Tuning Finished!")
    if results:
        sorted_results = sorted(results, key=lambda x: x['avg_loss'])
        best_params = sorted_results[0]['params']
        print("\n--- Results Summary (Sorted by Average Validation Loss) ---")
        for res in sorted_results:
            print(f"  - Avg Loss: {res['avg_loss']:.4f} | Params: {res['params']}")
        
        print("\n🏆 Best Hyperparameter Combination Found:")
        print(f"   - Average Validation Loss: {sorted_results[0]['avg_loss']:.4f}")
        for key, value in best_params.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print("\nACTION: Update these values as defaults in `run_assistant.py` and run one final training.")
    else:
        print("No successful runs were completed.")
    
    print("   Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("="*70)

if __name__ == '__main__':
    run_cv_tuning()