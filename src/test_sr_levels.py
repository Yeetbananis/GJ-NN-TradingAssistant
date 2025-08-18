import yfinance as yf
import mplfinance as mpf
from pathlib import Path
import pandas as pd

# Import both level-finding functions from your main script
from run_assistant import find_clustered_levels, find_volume_levels

print("--- Testing Hybrid S/R & Volume POC Detection ---")

# --- 1. Calculate Price-Based S/R Levels ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("\nDownloading 2 years of daily data for price analysis...")
df_daily_longterm = yf.download("GBPJPY=X", period="2y", interval="1d", auto_adjust=False, progress=False)
if not df_daily_longterm.empty:
    df_daily_longterm.columns = df_daily_longterm.columns.get_level_values(0)

major_zones = find_clustered_levels(df_daily_longterm)
long_term_levels = major_zones['level'].tolist() if not major_zones.empty else []

print("Finding recent short-term levels (weekly and monthly highs/lows)...")
last_7_days = df_daily_longterm.tail(7)
last_30_days = df_daily_longterm.tail(30)
short_term_levels = [
    last_7_days['High'].max(),
    last_7_days['Low'].min(),
    last_30_days['High'].max(),
    last_30_days['Low'].min()
]
price_based_levels = sorted(list(set(long_term_levels + short_term_levels)))
print(f"[OK] Found {len(price_based_levels)} unique price-based S/R levels.")


# --- 2. Calculate and Print Volume-Based POC Levels ---
print("\nCalculating multi-timeframe Volume POCs...")
poc_levels_dict = find_volume_levels()
volume_based_levels = sorted([v['price'] for v in poc_levels_dict.values() if v and v.get('price') is not None])
print(f"[OK] Found {len(volume_based_levels)} unique volume-based POC levels.")

# **NEW: Print the detected POC levels**
print("\n[Volume Points of Control (POC)]")
if volume_based_levels:
    for level in volume_based_levels:
        print(f"  - {level:.4f}")
else:
    print("  - None detected.")


# --- 3. Create Master List and Analyze Nearest Levels---
all_levels_combined = sorted(list(set(price_based_levels + volume_based_levels)))
cleaned_master_list = []
if all_levels_combined:
    cleaned_master_list.append(all_levels_combined[0])
    for i in range(1, len(all_levels_combined)):
        if abs(all_levels_combined[i] - cleaned_master_list[-1]) > 0.25:
            cleaned_master_list.append(all_levels_combined[i])

if not cleaned_master_list:
    print("\nNo S/R or POC levels were detected.")
else:
    current_price = df_daily_longterm['Close'].iloc[-1]
    print(f"\n--- Analysis based on Current Price: {current_price:.4f} ---")
    
    support_levels = [level for level in cleaned_master_list if level < current_price]
    resistance_levels = [level for level in cleaned_master_list if level > current_price]

    print("\n[Nearest Support Zones (Price or Volume)]")
    for level in support_levels[-2:]:
        print(f"  - {level:.4f}")
        
    print("\n[Nearest Resistance Zones (Price or Volume)]")
    for level in resistance_levels[:2]:
        print(f"  - {level:.4f}")

# --- 4. Plot Charts with Distinct Colors for Each Level Type ---
print("\nGenerating charts...")

# Create combined lists for all levels, colors, and line styles
all_hlines = price_based_levels + volume_based_levels
all_colors = ['b'] * len(price_based_levels) + ['r'] * len(volume_based_levels)
all_linestyle = ['--'] * len(price_based_levels) + [':'] * len(volume_based_levels)

daily_chart_df = df_daily_longterm.tail(252).copy()
ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
daily_chart_df[ohlc_cols] = daily_chart_df[ohlc_cols].apply(pd.to_numeric, errors='coerce')
daily_chart_df.dropna(inplace=True)

# Plot the long-term daily chart
mpf.plot(
    daily_chart_df, type='candle', style='yahoo',
    title='GBPJPY Daily Chart with Price S/R (Blue) and Volume POC (Red)',
    ylabel='Price',
    hlines=dict(hlines=all_hlines, colors=all_colors, linestyle=all_linestyle, alpha=0.6),
    figratio=(16, 8), savefig=PROJECT_ROOT / 'sr_and_poc_chart_daily.png'
)
print(f"[OK] Daily chart saved to 'sr_and_poc_chart_daily.png'")

# Plot the recent 4-Hour chart
print("Downloading recent 4-Hour data for clean chart...")
df_4h_recent = yf.download("GBPJPY=X", period="60d", interval="4h", auto_adjust=False, progress=False)
if not df_4h_recent.empty:
    df_4h_recent.columns = df_4h_recent.columns.get_level_values(0)
    df_4h_recent[ohlc_cols] = df_4h_recent[ohlc_cols].apply(pd.to_numeric, errors='coerce')
    df_4h_recent.dropna(inplace=True)

    mpf.plot(
        df_4h_recent, type='candle', style='yahoo',
        title='GBPJPY 4-Hour Chart with Price S/R (Blue) and Volume POC (Red)',
        ylabel='Price',
        hlines=dict(hlines=all_hlines, colors=all_colors, linestyle=all_linestyle, alpha=0.7),
        figratio=(16, 8), savefig=PROJECT_ROOT / 'sr_and_poc_chart_recent_4h.png'
    )
    print(f"[OK] Recent 4-Hour chart saved to 'sr_and_poc_chart_recent_4h.png'")