import pandas as pd

def compute_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Example usage:
# df['SMA_20'] = compute_sma(df, window=20)

def compute_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Example usage:
# df['EMA_12'] = compute_ema(df, window=12)

def compute_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Example usage:
# df['RSI_14'] = compute_rsi(df, window=14)

def compute_macd(data, short_window, long_window, signal_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

# Example usage:
# df['MACD_Line'], df['Signal_Line'], df['MACD_Histogram'] = compute_macd(df, short_window=12, long_window=26, signal_window=9)

def compute_bollinger_bands(data, window, num_std):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    return upper_band, lower_band

# Example usage:
# df['Upper_Band'], df['Lower_Band'] = compute_bollinger_bands(df, window=20, num_std=2)

def compute_stochastic_oscillator(data, window):
    highest_high = data['High'].rolling(window=window).max()
    lowest_low = data['Low'].rolling(window=window).min()

    k = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()  # 3 is the default smoothing period for %D

    return k, d

# Example usage:
# df['%K'], df['%D'] = compute_stochastic_oscillator(df, window=14)


def compute_ichimoku_cloud(data, conversion_line_window, base_line_window, leading_span_b_window):
    high_max = data['High'].rolling(window=conversion_line_window).max()
    low_min = data['Low'].rolling(window=conversion_line_window).min()
    conversion_line = (high_max + low_min) / 2

    high_max = data['High'].rolling(window=base_line_window).max()
    low_min = data['Low'].rolling(window=base_line_window).min()
    base_line = (high_max + low_min) / 2

    leading_span_a = ((conversion_line + base_line) / 2).shift(conversion_line_window)

    high_max = data['High'].rolling(window=leading_span_b_window).max()
    low_min = data['Low'].rolling(window=leading_span_b_window).min()
    leading_span_b = ((high_max + low_min) / 2).shift(conversion_line_window)

    lagging_span = data['Close'].shift(-conversion_line_window)

    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span

# Example usage:
# df['Conversion_Line'], df['Base_Line'], df['Leading_Span_A'], df['Leading_Span_B'], df['Lagging_Span'] = compute_ichimoku_cloud(df, conversion_line_window=9, base_line_window=26, leading_span_b_window=52)
