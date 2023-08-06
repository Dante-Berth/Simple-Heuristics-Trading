import tensorflow as tf
import numpy as np
from simple_heuritic_tradings.indicators.list_indicators import *
from simple_heuritic_tradings.utils.utils_df import  *
PATH = r"C:\Users\alexw\Documents\Git\AI-WORK\Advanced-Trading\simple_heuritic_tradings\Data\ETHUSDT-5m.zip"
dataframe = opener_dataframe(PATH)
windows = 2

@tf.keras.utils.register_keras_serializable()
class Signlog(tf.keras.layers.Layer):
    """
    Activation function from a paper Dreamer but adding a weight for increasing or reducing the input importance
    """
    def __init__(self,**kwargs):
        super(Signlog, self).__init__(**kwargs)
        self.weight = self.add_weight(
            name='weights',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-10, maxval=10),
            trainable=True
        )

    def get_config(self):
        config = super(Signlog, self).get_config()
        return config

    def call(self, inputs):
        return tf.math.sign(inputs) * tf.math.log(tf.keras.activations.relu(self.weight)*tf.math.abs(inputs) + 1)
class super_lstm(tf.keras.layers.Layer):
    def __init__(self):
        super(super_lstm, self).__init__()
        self.lstm_1 = tf.keras.layers.LSTM(units=64,return_sequences=True,activation="tanh")
        self.lstm_2 = tf.keras.layers.LSTM(units=32,return_sequences=True,activation="tanh")
        self.lstm_3 = tf.keras.layers.LSTM(units=16,return_sequences=True,activation="tanh")
        self.lstm_4 = tf.keras.layers.LSTM(units=8,return_sequences=True,activation="tanh")

        self.cnn_1 = tf.keras.layers.Conv1D(filters = 64, kernel_size=4,strides=2)
        self.cnn_2 = tf.keras.layers.Conv1D(filters=32, kernel_size=4,strides=2)
        self.cnn_3 = tf.keras.layers.Conv1D(filters=16 , kernel_size=2,strides=1)

        self.mlp_1 = tf.keras.layers.Dense(units = 64,activation="tanh")
        self.mlp_2 = tf.keras.layers.Dense(units=32, activation="tanh")
        self.mlp_3 = tf.keras.layers.Dense(units=16, activation="tanh")

        self.mlp_4 = tf.keras.layers.Dense(units=64)
        self.mlp_5 = tf.keras.layers.Dense(units=32)
        self.mlp_6 = tf.keras.layers.Dense(units=16)
        self.LSTM_SIZE = 32
        self.dropout = tf.keras.layers.Dropout(0.3)


        self.encoder_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))
        self.encoder_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))
        self.encoder_3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=False, activation='relu'))
        self.encoder_4 = tf.keras.layers.Dense(128,activation="linear")

        self.decoder_1 = tf.keras.layers.RepeatVector(32)
        self.decoder_2 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.decoder_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32))
        self.decoder_4 = tf.keras.layers.Flatten()
        self.decoder_5 = tf.keras.layers.Dense(1)
        self.sign_log_1 = Signlog()
        self.sign_log_2 = Signlog()
        self.sign_log_3 = Signlog()

        self.mlp_7 = tf.keras.layers.Dense(1)
    def call(self,input):
        x_1 = self.lstm_1(input)
        x_1 = self.lstm_2(x_1)
        x_1 = self.lstm_3(x_1)
        x_1 = tf.keras.layers.Flatten()(x_1)

        x_2 = self.cnn_1(input)
        x_2 = self.dropout(self.cnn_2(x_2))
        x_2 = self.dropout(self.cnn_3(x_2))
        x_2 = tf.keras.layers.Flatten()(x_2)

        x_3 = self.mlp_1(input)
        x_3 = self.dropout(self.mlp_2(x_3))
        x_3 = self.dropout(self.mlp_3(x_3))
        x_3 = tf.keras.layers.Flatten()(x_3)

        x_4 = tf.keras.layers.Concatenate()([x_1,x_2,x_3])
        x_4 = self.sign_log_1(self.mlp_4(x_4))
        x_4 = self.sign_log_2(self.mlp_5(x_4))
        x_4 = self.sign_log_3(self.mlp_6(x_4))
        x_4 = self.mlp_7(x_4)

        y_1 = self.encoder_1(input)
        y_1 = self.encoder_2(y_1)
        y_1 = self.encoder_3(y_1)
        y_1 = self.encoder_4(y_1)

        y_2 = self.decoder_1(y_1)
        y_2 = self.decoder_2(y_2)
        y_2 = self.decoder_3(y_2)
        y_2 = self.decoder_4(y_2)
        y_2 = self.decoder_5(y_2)



        x_5 = x_4 + y_2
        return x_5


dataframe[f'SMA_{windows}'] = compute_sma(dataframe,windows)
dataframe[f'EMA_{windows}'] = compute_ema(dataframe,windows)
dataframe[f'RSI_{windows}'] = compute_rsi(dataframe, windows)
dataframe['MACD_Line'], dataframe['Signal_Line'], dataframe['MACD_Histogram'] = compute_macd(dataframe, short_window=12, long_window=26, signal_window=9)
dataframe['Upper_Band'], dataframe['Lower_Band'] = compute_bollinger_bands(dataframe, window=windows, num_std=2)


columns = ['open_price',f'SMA_{windows}',f'EMA_{windows}']
data_x = dataframe[columns]
print(data_x.head(5))
exit()
# Normalization
for column in columns:
    data_x[column] = (data_x[column] - data_x[column].shift(-1))/(data_x[column].shift(-1))*100
print(data_x.head())
#Creation des dataframes
X = []
Y = []
window_size = 32
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

history = model.fit(x=X_train, y=Y_train, epochs=200,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks, verbose=1, shuffle=False, batch_size = 64)