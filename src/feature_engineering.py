# In src/feature_engineering.py

import pandas as pd
from scipy.signal import find_peaks

def find_swing_points(series, order=5):
    """Finds swing highs and lows in a price series."""
    # Find peaks (swing highs)
    high_peaks_indices, _ = find_peaks(series, distance=order)
    # Find troughs (swing lows) by inverting the series
    low_peaks_indices, _ = find_peaks(-series, distance=order)
    return high_peaks_indices, low_peaks_indices

def find_fair_value_gaps(df):
    """Identifies Fair Value Gaps (FVGs)."""
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG: Gap between candle i-2 high and candle i low
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            fvgs.append({
                'type': 'bullish',
                'top': df['low'].iloc[i],
                'bottom': df['high'].iloc[i-2],
                'timestamp': df.index[i]
            })
        # Bearish FVG: Gap between candle i-2 low and candle i high
        if df['low'].iloc[i-2] > df['high'].iloc[i]:
             fvgs.append({
                'type': 'bearish',
                'top': df['low'].iloc[i-2],
                'bottom': df['high'].iloc[i],
                'timestamp': df.index[i]
            })
    return fvgs

# At any given time `t`, we can run these functions on recent data
# to get a list of support (bullish FVGs, swing lows) and
# resistance (bearish FVGs, swing highs) zones.