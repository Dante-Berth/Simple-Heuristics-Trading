import numpy as np
from simple_heuritic_tradings.indicators.list_indicators import *
from simple_heuritic_tradings.utils.utils_df import *
from simple_heuritic_tradings.model.model_lstm import super_lstm
import tensorflow as tf
from simple_heuritic_tradings.layer.dain import DAIN_Layer
from scipy import signal
PATH = r"C:\Users\alexw\Documents\Git\AI-WORK\Advanced-Trading\simple_heuritic_tradings\Data\ETHUSDT-5m.zip"
dataframe = opener_dataframe(PATH)
PATH_2 = r"C:\Users\alexw\Documents\Git\AI-WORK\Advanced-Trading\simple_heuritic_tradings\Data\ETHUSDT-5m-without-aggr.zip"
dataframe_2 = opener_dataframe(PATH_2)

dataframe = pd.merge(dataframe, dataframe_2, on='open_date', how='inner')

"""
windows = 2
windows_1 = 12
windows_2 = 16
dataframe[f'SMA_{windows}'] = compute_sma(dataframe,windows)
dataframe[f'EMA_{windows}'] = compute_ema(dataframe,windows)
dataframe[f'SMA_{windows_2}'] = compute_rsi(dataframe, windows_2)
dataframe[f'EMA_{windows_1}'] = compute_rsi(dataframe, windows_1)
dataframe[f'EMA_{windows_2}'] = compute_rsi(dataframe, windows_2)
dataframe['MACD_Line'], dataframe['Signal_Line'], dataframe['MACD_Histogram'] = compute_macd(dataframe, short_window=12, long_window=26, signal_window=9)
dataframe['Upper_Band'], dataframe['Lower_Band'] = compute_bollinger_bands(dataframe, window=windows, num_std=2)
"""

columns = ['open_price_x',"high","low"]
data_x = dataframe[columns]
print(data_x.head(5))

# Normalization

# data_x = (data_x - data_x.shift(1))/(data_x.shift(1))*100

print(data_x.head())
from ta.momentum import KAMAIndicator
dataframe['close_ema'] = dataframe['close_x'].ewm(span=1, adjust=False).mean()
dataframe["KAMAIndicator"] = KAMAIndicator(dataframe["close_ema"]).kama()

dataframe['high_ema'] = dataframe['high'].ewm(span=1, adjust=False).mean()
dataframe['low_ema'] = dataframe['low'].ewm(span=1, adjust=False).mean()

from ta.momentum import AwesomeOscillatorIndicator
dataframe["AwesomeOscillatorIndicator"] = AwesomeOscillatorIndicator(high=dataframe['high_ema'],low=dataframe['low_ema']).awesome_oscillator()

from ta.volatility import BollingerBands
indicator_bb = BollingerBands(close=dataframe["low_ema"],window=20,window_dev=2)
dataframe['bb_bbm'] = indicator_bb.bollinger_mavg()
dataframe['bb_bbh'] = indicator_bb.bollinger_hband()
dataframe['bb_bbl'] = indicator_bb.bollinger_lband()

columns = ['open_price_x',"high","low","AwesomeOscillatorIndicator","bb_bbm",'bb_bbh','bb_bbl',"KAMAIndicator"]
data_x = dataframe[columns]

#Creation des dataframes
X = []
Y = []
window_size = 32
b, a = signal.butter(8, 0.125)
from tqdm import tqdm
for i in tqdm(range(window_size+10,len(data_x)-100)):#500)): #
    if (data_x.iloc[i-window_size:i]).isnull().values.any() == False:
        windows = data_x.iloc[i-window_size:i].to_numpy()
        open_price_signal = data_x["open_price_x"].iloc[i - window_size:i].values
        high_price_signal = data_x["high"].iloc[i - window_size:i].values
        low_price_signal = data_x["low"].iloc[i - window_size:i].values
        price_kamai = data_x["KAMAIndicator"].iloc[i - window_size:i].values
        price_awesome_oscillator = data_x["AwesomeOscillatorIndicator"].iloc[i - window_size:i].values
        price_bb_bbm = data_x["bb_bbm"].iloc[i - window_size:i].values
        price_bb_bbh = data_x["bb_bbh"].iloc[i - window_size:i].values
        price_bb_bbl = data_x["bb_bbl"].iloc[i - window_size:i].values
        open_price_new_signal = signal.filtfilt(b, a, open_price_signal, padlen=window_size-1)
        high_price_new_signal = signal.filtfilt(b, a, high_price_signal, padlen=window_size - 1)
        low_price_new_signal = signal.filtfilt(b, a, low_price_signal, padlen=window_size - 1)
        concatenated_array = np.column_stack((open_price_signal, high_price_signal,low_price_signal,
                                              price_kamai, price_awesome_oscillator, price_bb_bbm, price_bb_bbh,
                                              price_bb_bbl,
                                              low_price_new_signal,high_price_new_signal,open_price_new_signal))
        X.append(np.asarray(concatenated_array))
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
dain_layer = DAIN_Layer(mode='full', input_dim=np.shape(X_train[0])[-1])
inputs = tf.keras.layers.Input(shape=np.shape(X_train[0]))
x = dain_layer(inputs)
outputs = super_lstm()(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# print the network
model.summary()
model.compile(optimizer='adam', loss='mse')

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)  # early stopping to prevent overfitting

validation_data=(X_test, Y_test)
import plotly.express as px
callbacks = [es]
epochs = 7
dict_test = {"X_test":X_test,"Y_test":Y_test}
nb_test = len(X_test)
import plotly.graph_objects as go
for e in range(epochs):
    history = model.fit(x=X_train, y=Y_train,verbose = 1,batch_size=64)
    model.evaluate(x=X_test, y=Y_test)
    Y_pred =[]
    Y_true = []
    for i in tqdm(range(nb_test//10)):
        Y_pred.append(float(model.predict(np.expand_dims(dict_test["X_test"][i], axis=0),verbose=0)[0,0]))
        Y_true.append(dict_test["Y_test"][i])

    fig = go.Figure()
    t = np.linspace(0,1,nb_test)
    fig.add_trace(go.Scatter(x=t,
                             y=Y_pred,
                             mode='lines',
                             name='Prediction Signal'))

    fig.add_trace(go.Scatter(x=t,
                             y=Y_true,
                             mode='lines',
                             name='True Signal'))


    fig.update_layout(title=f'Comparison of Predicted and True Signals ',
                      xaxis_title='Time Index',
                      yaxis_title='Y',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.show()

# print the network
model.summary()
model.compile(optimizer='adam', loss='mae')

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)  # early stopping to prevent overfitting

validation_data=(X_test, Y_test)
import plotly.express as px
callbacks = [es]
epochs = 7
dict_test = {"X_test":X_test,"Y_test":Y_test}
nb_test = len(X_test)
for e in range(epochs):
    history = model.fit(x=X_train, y=Y_train,verbose = 1,batch_size=64)
    model.evaluate(x=X_test, y=Y_test)
    Y_pred =[]
    Y_true = []
    for i in tqdm(range(nb_test//10)):
        Y_pred.append(float(model.predict(np.expand_dims(dict_test["X_test"][i], axis=0),verbose=0)[0,0]))
        Y_true.append(dict_test["Y_test"][i])

    fig = go.Figure()
    t = np.linspace(0,1,nb_test)
    fig.add_trace(go.Scatter(x=t,
                             y=Y_pred,
                             mode='lines',
                             name='Prediction Signal'))

    fig.add_trace(go.Scatter(x=t,
                             y=Y_true,
                             mode='lines',
                             name='True Signal'))


    fig.update_layout(title=f'Comparison of Predicted and True Signals ',
                      xaxis_title='Time Index,MAE',
                      yaxis_title='Y',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.show()

model.compile(optimizer='adam', loss=tf.keras.losses.LogCosh())

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)  # early stopping to prevent overfitting

validation_data=(X_test, Y_test)
import plotly.express as px
callbacks = [es]
epochs = 7
dict_test = {"X_test":X_test,"Y_test":Y_test}
nb_test = len(X_test)
for e in range(epochs):
    history = model.fit(x=X_train, y=Y_train,verbose = 1,batch_size=64)
    model.evaluate(x=X_test, y=Y_test)
    Y_pred =[]
    Y_true = []
    for i in tqdm(range(nb_test//10)):
        Y_pred.append(float(model.predict(np.expand_dims(dict_test["X_test"][i], axis=0),verbose=0)[0,0]))
        Y_true.append(dict_test["Y_test"][i])

    fig = go.Figure()
    t = np.linspace(0,1,nb_test)
    fig.add_trace(go.Scatter(x=t,
                             y=Y_pred,
                             mode='lines',
                             name='Prediction Signal'))

    fig.add_trace(go.Scatter(x=t,
                             y=Y_true,
                             mode='lines',
                             name='True Signal'))


    fig.update_layout(title=f'Comparison of Predicted and True Signals ',
                      xaxis_title='Time Index,LogCosh',
                      yaxis_title='Y',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.show()