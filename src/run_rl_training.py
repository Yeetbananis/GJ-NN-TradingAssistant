import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import pytz
import yfinance as yf
from stable_baselines3 import PPO
import warnings

# =============================================================================
# ALL-IN-ONE REINFORCEMENT LEARNING TRAINING SCRIPT
# =============================================================================

# --- 1. Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = PROJECT_ROOT / "data/labels/manual_labels.csv"
RL_MODEL_SAVE_PATH = PROJECT_ROOT / "models/rl_trade_manager.zip"
TOTAL_TIMESTEPS = 100_000

# --- 2. Helper Functions for Data ---

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

# --- 3. The Trading Environment Class ---

class TradingEnv(gym.Env):
    def __init__(self, feature_df, label_df, max_steps=288):
        super(TradingEnv, self).__init__()
        self.df = feature_df
        self.labels = label_df.dropna(subset=['entry_price', 'sl_price', 'max_reward_ratio']).copy()
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2) # 0: HOLD, 1: CLOSE
        num_market_features = len(self.df.columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_market_features + 3,), dtype=np.float32)

    def _get_state(self):
        market_features = self.df.iloc[self.current_step].values
        position_status = 1.0 if self.in_trade else 0.0
        time_in_trade_norm = self.steps_in_trade / self.max_steps
        current_price = self.df['close'].iloc[self.current_step]
        pnl_in_r = ((current_price - self.entry_price) * self.trade_direction) / self.risk_per_share if self.in_trade and self.risk_per_share > 0 else 0.0
        trade_features = np.array([position_status, time_in_trade_norm, pnl_in_r])
        return np.concatenate([market_features, trade_features]).astype(np.float32)

    def _calculate_reward(self, terminated):
        current_price = self.df['close'].iloc[self.current_step]
        pnl_in_r = ((current_price - self.entry_price) * self.trade_direction) / self.risk_per_share if self.risk_per_share > 0 else 0.0
        if terminated:
            if pnl_in_r >= self.take_profit_r: return 1.0
            elif pnl_in_r <= -1.0: return -1.5
            else: return pnl_in_r
        else:
            return pnl_in_r * 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ** THE GUARANTEED FIX: A robust loop to find a valid starting point **
        while True:
            random_trade = self.labels.sample(n=1).iloc[0]
            entry_price_label = random_trade['entry_price']
            label_date = pd.to_datetime(random_trade['date']).date()
            session_features = self.df[(self.df.index.date == label_date) & (self.df['is_trading_session'] == 1)]
            if not session_features.empty:
                with warnings.catch_warnings(): # Suppress the FutureWarning from idxmin
                    warnings.simplefilter("ignore", FutureWarning)
                    start_timestamp = (session_features['close'] - entry_price_label).abs().idxmin()
                self.current_step = self.df.index.get_loc(start_timestamp)
                if self.current_step + self.max_steps < len(self.df):
                    break # Found a valid start, exit the loop
        
        self.in_trade = True
        self.steps_in_trade = 0
        self.entry_price = self.df['close'].iloc[self.current_step]
        self.trade_direction = random_trade['trade_direction']
        sl_price_label = random_trade['sl_price']
        self.risk_per_share = abs(self.entry_price - sl_price_label)
        self.take_profit_r = random_trade['max_reward_ratio'] if pd.notna(random_trade['max_reward_ratio']) and random_trade['max_reward_ratio'] > 0 else 2.0
        return self._get_state(), {}

    def step(self, action):
        self.current_step += 1; self.steps_in_trade += 1
        terminated, truncated = False, False
        current_high = self.df['high'].iloc[self.current_step]
        current_low = self.df['low'].iloc[self.current_step]
        if self.trade_direction == 1:
            if current_low <= self.entry_price - self.risk_per_share: terminated = True
            elif current_high >= self.entry_price + (self.risk_per_share * self.take_profit_r): terminated = True
        else:
            if current_high >= self.entry_price + self.risk_per_share: terminated = True
            elif current_low <= self.entry_price - (self.risk_per_share * self.take_profit_r): terminated = True
        if action == 1: terminated = True
        if self.steps_in_trade >= self.max_steps: truncated = True
        if terminated or truncated: self.in_trade = False
        reward = self._calculate_reward(terminated)
        state = self._get_state()
        return state, reward, terminated, truncated, {}

# --- 4. Main Training Execution ---
if __name__ == '__main__':
    print("--- Starting Reinforcement Learning Agent Training ---")
    df_1h, df_5m = load_and_preprocess_data()
    if df_1h is None or df_5m is None:
        print("Could not load data. Exiting.")
    else:
        feature_data = add_key_level_features(df_5m, df_1h)
        label_data = pd.read_csv(LABEL_FILE)
        
        print("Creating trading environment...")
        env = TradingEnv(feature_df=feature_data, label_df=label_data)
        
        print("Creating PPO agent...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(PROJECT_ROOT / "rl_logs/"))
        
        print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        print("\nTraining complete.")
        RL_MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
        model.save(RL_MODEL_SAVE_PATH)
        print(f"RL Agent saved to {RL_MODEL_SAVE_PATH}")