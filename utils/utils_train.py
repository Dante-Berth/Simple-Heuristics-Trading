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

#Creation des dataframes
X = []
Y = []
window_size = 32
b, a = signal.butter(8, 0.125)
from tqdm import tqdm
for i in tqdm(range(window_size+10,len(data_x)-100)):#500)): #
    if (data_x.iloc[i-window_size:i]).isnull().values.any() == False:
        windows = data_x.iloc[i-window_size:i].to_numpy()
        open_price_signal = dataframe["open_price_x"].iloc[i - window_size:i].values
        high_price_signal = dataframe["high"].iloc[i - window_size:i].values
        low_price_signal = dataframe["low"].iloc[i - window_size:i].values
        open_price_new_signal = signal.filtfilt(b, a, open_price_signal, padlen=window_size-1)
        high_price_new_signal = signal.filtfilt(b, a, high_price_signal, padlen=window_size - 1)
        low_price_new_signal = signal.filtfilt(b, a, low_price_signal, padlen=window_size - 1)
        concatenated_array = np.column_stack((open_price_signal, high_price_signal,low_price_signal,low_price_new_signal,high_price_new_signal,open_price_new_signal))
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


# https://stackoverflow.com/questions/67940697/hessian-matrix-of-a-keras-model-with-tf-hessians
class FisherInformationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Create a GradientTape to watch the trainable variables
        with tf.GradientTape(persistent=True) as hessian_tape:
            # Another GradientTape to compute the gradients
            with tf.GradientTape() as gradient_tape:
                gradient_tape.watch(self.model.trainable_variables)
                # Compute loss
                predictions = self.model(self.validation_data[0])
                loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.validation_data[1], predictions))

            # Compute first-order gradients
            gradients = gradient_tape.gradient(loss, self.model.trainable_variables)

        # Compute second-order gradients (Hessian)
        hessian = [hessian_tape.jacobian(g, w) for g, w in zip(gradients, self.model.trainable_variables)]

        # Fisher Information is the negative expectation of the Hessian of the log-likelihood.
        # In practice, you would need to multiply by -1 and take the expectation over your data distribution
        fisher_information = [-h for h in hessian]

        # You can then log, save, or otherwise use the 'fisher_information'
        # For example, print the Fisher Information for the first layer
        print("Fisher Information for first layer at epoch", epoch, ":", fisher_information[0])



# Usage remains the same
validation_data=(X_test, Y_test)
#fisher_callback = FisherInformationCallback(validation_data)
callbacks = [es]

history = model.fit(x=X_train, y=Y_train, epochs=10,
                    validation_data=validation_data,
                    callbacks=callbacks, verbose=1, shuffle=False, batch_size = 32)
Y_pred_test = model.predict(X_test)
