import tensorflow as tf
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

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = tf.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.layers.Dropout(dropout)(x)
    x = tf.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = tf.layers.Dropout(dropout)(x)
    x = tf.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
class super_lstm(tf.keras.layers.Layer):
    def __init__(self):
        super(super_lstm, self).__init__()
        self.lstm_1 = tf.keras.layers.LSTM(units=64,return_sequences=True,activation="tanh")
        self.lstm_2 = tf.keras.layers.LSTM(units=32,return_sequences=True,activation="tanh")
        self.lstm_3 = tf.keras.layers.LSTM(units=16,return_sequences=True,activation="tanh")
        self.lstm_4 = tf.keras.layers.LSTM(units=16,return_sequences=True,activation="tanh")

        self.cnn_1 = tf.keras.layers.Conv1D(filters = 64, kernel_size=4,strides=2)
        self.cnn_2 = tf.keras.layers.Conv1D(filters=32, kernel_size=4,strides=2)
        self.cnn_3 = tf.keras.layers.Conv1D(filters=8 , kernel_size=2,strides=1)

        self.mlp_1 = tf.keras.layers.Dense(units = 64,activation="tanh")
        self.mlp_2 = tf.keras.layers.Dense(units=32, activation="tanh")
        self.mlp_3 = tf.keras.layers.Dense(units=32, activation="tanh")

        self.mlp_4 = tf.keras.layers.Dense(units=64)
        self.mlp_5 = tf.keras.layers.Dense(units=32)
        self.mlp_6 = tf.keras.layers.Dense(units=32)
        self.LSTM_SIZE = 32
        self.dropout = tf.keras.layers.Dropout(0.3)


        self.encoder_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))
        self.encoder_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))
        self.encoder_3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=False, activation='relu'))
        self.encoder_4 = tf.keras.layers.Dense(128,activation="linear")

        self.decoder_1 = tf.keras.layers.RepeatVector(32)
        self.decoder_2 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.decoder_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))
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

