import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
import pytz
import yfinance as yf
import talib
import shap

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class TradeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = {k: torch.tensor(v, dtype=torch.long if k in ['pattern_state', 'entry_signal', 'setup_quality'] else torch.float32).to(device) for k, v in labels.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.labels.items()}

# MultiStageLSTM Model
class MultiStageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(MultiStageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stage 1: Pattern State
        self.pattern_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.pattern_fc = nn.Linear(hidden_size, 3)  # none, building, present

        # Stage 2: Entry Signal
        self.signal_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.signal_fc = nn.Linear(hidden_size, 4)  # none, BOS_15m_up, BOS_15m_down, reversal_15m

        # Stage 3: Multi-task Rating
        self.rating_lstm = nn.LSTM(input_size + 7, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.rating_fc_direction = nn.Linear(hidden_size, 1)  # binary direction
        self.rating_fc_quality = nn.Linear(hidden_size, 5)  # 1-5 quality
        self.rating_fc_rr = nn.Linear(hidden_size, 1)  # R:R regression
        self.rating_fc_sl = nn.Linear(hidden_size, 1)  # SL% probability

        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, stage=1):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if stage == 1:
            out, _ = self.pattern_lstm(x, (h0, c0))
            out = self.bn(out[:, -1, :])
            out = self.relu(out)
            return self.pattern_fc(out)
        elif stage == 2:
            out, _ = self.signal_lstm(x, (h0, c0))
            out = self.bn(out[:, -1, :])
            out = self.relu(out)
            return self.signal_fc(out)
        else:  # stage 3
            out, _ = self.rating_lstm(x, (h0, c0))
            out = self.bn(out[:, -1, :])
            out = self.relu(out)
            direction = self.sigmoid(self.rating_fc_direction(out))
            quality = self.rating_fc_quality(out)
            rr = self.rating_fc_rr(out)
            sl = self.sigmoid(self.rating_fc_sl(out))
            return direction, quality, rr, sl

# Focal Loss for imbalanced classes
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return self.alpha * (1 - pt) ** self.gamma * ce_loss.mean()

# Data Preprocessing
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Entry_Time'], date_format='%Y-%m-%d %H:%M:%S%z')
    df = df.dropna()

    # Encode categorical columns
    le_pattern = LabelEncoder()
    le_signal = LabelEncoder()
    df['Pattern_State'] = le_pattern.fit_transform(df['Pattern_State'])
    df['Entry_Signal'] = le_signal.fit_transform(df['Entry_Signal'])

    # Features and labels
    sequence_cols = [col for col in df.columns if col.startswith('t-') and col not in ['t-0_Entry_Signal', 't-0_Pattern_State']]
    trade_cols = ['Dist_to_Zone', 'Zone_Swept', 'Zone_Price']
    features = df[sequence_cols + trade_cols].values
    labels = {
        'pattern_state': df['Pattern_State'].values,
        'entry_signal': df['Entry_Signal'].values,
        'setup_quality': df['Setup_Quality'].values,
        'expected_rr': df['Expected_RR'].values if 'Expected_RR' in df else np.zeros(len(df)),
        'win_probability': df['Win_Probability'].values if 'Win_Probability' in df else np.zeros(len(df))
    }

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, stratify=labels['pattern_state'], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, {k: v for k, v in y_temp.items()}, test_size=0.5, stratify=y_temp['pattern_state'], random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, le_pattern, le_signal

# Training Function
def train_stage(model, dataloader, optimizer, criterion, stage, epochs=200, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            if stage == 3:
                out = model(batch_x, stage=stage)
                loss = (criterion['quality'](out[1], batch_y['setup_quality']) * 0.4 +
                        nn.MSELoss()(out[2], batch_y['expected_rr'].unsqueeze(1)) * 0.3 +
                        nn.BCELoss()(out[3], batch_y['win_probability'].unsqueeze(1)) * 0.3)
            else:
                out = model(batch_x, stage=stage)
                loss = criterion[str(stage)](out, batch_y['pattern_state' if stage == 1 else 'entry_signal'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Stage {stage}, Loss: {avg_loss:.4f}')
        
        # Early stopping
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in dataloader:
                if stage == 3:
                    out = model(batch_x, stage=stage)
                    val_loss += (criterion['quality'](out[1], batch_y['setup_quality']) * 0.4 +
                                 nn.MSELoss()(out[2], batch_y['expected_rr'].unsqueeze(1)) * 0.3 +
                                 nn.BCELoss()(out[3], batch_y['win_probability'].unsqueeze(1)) * 0.3).item()
                else:
                    out = model(batch_x, stage=stage)
                    val_loss += criterion[str(stage)](out, batch_y['pattern_state' if stage == 1 else 'entry_signal']).item()
            val_loss /= len(dataloader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'multistage_lstm_stage{stage}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

# Main Training Pipeline
def main():
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, le_pattern, le_signal = preprocess_data('data/enhanced_manual_data.csv')

    # Create DataLoaders
    train_dataset = TradeDataset(X_train, y_train)
    val_dataset = TradeDataset(X_val, y_val)
    test_dataset = TradeDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = MultiStageLSTM(input_size).to(device)

    # Define loss functions and optimizers
    criterion = {
        '1': nn.CrossEntropyLoss(),
        '2': FocalLoss(),
        'quality': nn.CrossEntropyLoss()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train each stage
    print("Training Stage 1: Pattern State")
    train_stage(model, train_loader, optimizer, criterion, stage=1)
    model.load_state_dict(torch.load('multistage_lstm_stage1.pth'))

    print("Training Stage 2: Entry Signal")
    train_stage(model, train_loader, optimizer, criterion, stage=2)
    model.load_state_dict(torch.load('multistage_lstm_stage2.pth'))

    print("Training Stage 3: Rating")
    train_stage(model, train_loader, optimizer, criterion, stage=3)

    # Save final model
    torch.save(model.state_dict(), 'gbpjpy_multistage_lstm.pth')

    # Save scaler and encoders
    with open('scaler.pkl', 'wb') as f:
        import pickle
        pickle.dump(scaler, f)
    with open('le_pattern.pkl', 'wb') as f:
        pickle.dump(le_pattern, f)
    with open('le_signal.pkl', 'wb') as f:
        pickle.dump(le_signal, f)

if __name__ == "__main__":
    main()