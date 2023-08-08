import numpy as np
from simple_heuritic_tradings.indicators.list_indicators import *
from simple_heuritic_tradings.utils.utils_df import *
from simple_heuritic_tradings.model.model_lstm import super_lstm
PATH = r"C:\Users\alexw\Documents\Git\AI-WORK\Advanced-Trading\simple_heuritic_tradings\Data\ETHUSDT-5m.zip"
dataframe = opener_dataframe(PATH)
windows = 2

dataframe[f'SMA_{windows}'] = compute_sma(dataframe,windows)
dataframe[f'EMA_{windows}'] = compute_ema(dataframe,windows)
dataframe[f'RSI_{windows}'] = compute_rsi(dataframe, windows)
dataframe['MACD_Line'], dataframe['Signal_Line'], dataframe['MACD_Histogram'] = compute_macd(dataframe, short_window=12, long_window=26, signal_window=9)
dataframe['Upper_Band'], dataframe['Lower_Band'] = compute_bollinger_bands(dataframe, window=windows, num_std=2)


columns = ['open_price',f'SMA_{windows}',f'EMA_{windows}']
data_x = dataframe[columns]
print(data_x.head(5))

# Normalization

data_x = (data_x - data_x.shift(-1))/(data_x.shift(-1))*100

print(data_x.head())

#Creation des dataframes
X = []
Y = []
window_size = 64
from tqdm import tqdm
for i in tqdm(range(window_size+10,len(data_x)-1000)):
    if (data_x.iloc[i-window_size:i]).isnull().values.any() == False:
        X.append(np.asarray(data_x.iloc[i-window_size:i].to_numpy()))
        Y.append(np.asarray(dataframe["5m_rebuilt_signal_sym7_12_4_Y_area_24_2.5"].iloc[i]))

X = np.array(X)
Y = np.array(Y)
testAndValid = 0.1
SPLIT = int(testAndValid*len(X))
X_train = X[:-SPLIT]
X_test = X[-SPLIT:]
Y_train = Y[:-SPLIT]
Y_test = Y[-SPLIT:]
lstm = super_lstm()
inputs = tf.keras.layers.Input(shape=np.shape(X_train[0]))
outputs = super_lstm()(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# print the network
model.summary()
model.compile(optimizer='adam', loss='mse')

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)  # early stopping to prevent overfitting
callbacks = [es]

history = model.fit(x=X_train, y=Y_train, epochs=2,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks, verbose=1, shuffle=False, batch_size = 64)