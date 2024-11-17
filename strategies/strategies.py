from simple_heuritic_tradings.indicators.list_indicators import *
from simple_heuritic_tradings.utils.utils_df import  *
PATH = None
dataframe = opener_dataframe(PATH)
windows = 2
dataframe[f'SMA_{windows}'] = compute_sma(dataframe,windows)
dataframe[f'EMA_{windows}'] = compute_ema(dataframe,windows)
dataframe[f'RSI_{windows}'] = compute_rsi(dataframe, windows)
dataframe['MACD_Line'], dataframe['Signal_Line'], dataframe['MACD_Histogram'] = compute_macd(dataframe, short_window=12, long_window=26, signal_window=9)
dataframe['Upper_Band'], dataframe['Lower_Band'] = compute_bollinger_bands(dataframe, window=2, num_std=2)


print(dataframe.head())
