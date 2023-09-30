import tensorflow as tf
from simple_heuritic_tradings.model.tsmixer import tsmixer_res_block
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


class transformer_encoder(tf.keras.layers.Layer):
    def __init__(self,head_size,num_heads,ff_dim,dropout=0.15,*args,**kwargs):
        super(transformer_encoder, self).__init__(*args,**kwargs)
        self.multiheadattention = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
                )
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.layernormalization_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.feedforwardconv1d_1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1,activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.feedforwardconv1d_2 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1,activation="relu")
        self.layernormalization_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, *args, **kwargs):
        x = self.multiheadattention(inputs,inputs)
        x = self.dropout_1(x)
        x = self.layernormalization_1(x)
        res = x + inputs
        x = self.feedforwardconv1d_1(x)
        x = self.dropout_2(x)
        x = self.feedforwardconv1d_2(x)
        x = self.layernormalization_2(x)
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
        self.mlp_8 = tf.keras.layers.Dense(16,activation="tanh")
        self.mlp_9 = tf.keras.layers.Dense(1)

        self.transformer_encoder_1 = transformer_encoder(num_heads=6,head_size=4,ff_dim=6)
        self.transformer_encoder_2 = transformer_encoder(num_heads=6, head_size=4, ff_dim=6)
        self.transformer_encoder_3 = transformer_encoder(num_heads=6, head_size=4, ff_dim=6)

        self.cnn_1_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2)
        self.cnn_2_2 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2)
        self.cnn_3_2 = tf.keras.layers.Conv1D(filters=8, kernel_size=2, strides=1)

        self.tsmixer_res_block_1 = tsmixer_res_block(activation="gelu", dropout=0.3, ff_dim=16)
        self.tsmixer_res_block_2 = tsmixer_res_block(activation="gelu", dropout=0.3, ff_dim=8)
        self.tsmixer_res_block_3 = tsmixer_res_block(activation="gelu", dropout=0.3, ff_dim=4)

        self.cnn_1_3 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2)
        self.cnn_2_3 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2)
        self.cnn_3_3 = tf.keras.layers.Conv1D(filters=8, kernel_size=2, strides=1)

        self.mlp_10 = tf.keras.layers.Dense(1)

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

        x_transformer = self.transformer_encoder_1(input)
        x_transformer = self.transformer_encoder_2(x_transformer)
        x_transformer = self.transformer_encoder_3(x_transformer)
        x_transformer = self.cnn_1_2(x_transformer)
        x_transformer = self.cnn_2_2(x_transformer)
        x_transformer = self.cnn_3_2(x_transformer)
        x_transformer = tf.keras.layers.Flatten()(x_transformer)
        x_transformer = self.mlp_9(x_transformer)

        x_tsmixer = self.tsmixer_res_block_1(input)
        x_tsmixer = self.tsmixer_res_block_2(x_tsmixer)
        x_tsmixer = self.tsmixer_res_block_3(x_tsmixer)
        x_tsmixer = self.cnn_1_3(x_tsmixer)
        x_tsmixer = self.cnn_2_3(x_tsmixer)
        x_tsmixer = self.cnn_3_3(x_tsmixer)
        x_tsmixer = tf.keras.layers.Flatten()(x_tsmixer)
        x_tsmixer = self.mlp_10(x_tsmixer)

        x_5 = x_4 + y_2 + x_transformer + x_tsmixer
        return x_5

