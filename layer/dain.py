import tensorflow as tf
@tf.keras.utils.register_keras_serializable(name="DAIN_Layer")
class DAIN_Layer(tf.keras.layers.Layer):
    def __init__(self, mode='adaptive_avg', input_dim=144,**kwargs):
        super(DAIN_Layer, self).__init__(**kwargs)

        self.mode = mode
        self.input_dim = input_dim
        # Parameters for adaptive average
        self.mean_layer = tf.keras.layers.Dense(self.input_dim, use_bias=False)


        # Parameters for adaptive std
        self.scaling_layer = tf.keras.layers.Dense(self.input_dim, use_bias=False)

        # Parameters for adaptive scaling
        self.gating_layer = tf.keras.layers.Dense(self.input_dim)

        self.eps = 1e-8

    def call(self, x):
        # Expecting  (batch_size, n_feature_vectors, dim)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = tf.reduce_mean(x, axis=-1, keepdims=True)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = tf.reduce_mean(x, axis=-1, keepdims=True)
            adaptive_avg = self.mean_layer(avg)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':


            # Step 1:
            avg = tf.reduce_mean(x, axis=-1, keepdims=True)
            adaptive_avg = self.mean_layer(avg)
            x = x - adaptive_avg

            # Step 2:
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            std = tf.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = tf.where(adaptive_std <= self.eps, 1.0, adaptive_std)
            x = x / adaptive_std

        elif self.mode == 'full':

            # Step 1:
            avg = tf.reduce_mean(x, axis=-1, keepdims=True)
            adaptive_avg = self.mean_layer(avg)
            x = x - adaptive_avg

            # Step 2:
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            std = tf.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = tf.where(adaptive_std <= self.eps, 1.0, adaptive_std)
            x = x / adaptive_std

            # Step 3:
            avg = tf.reduce_mean(x, axis=-1, keepdims=True)
            gate = tf.math.sigmoid(self.gating_layer(avg))
            x = x * gate

        else:
            assert False

        return x

    def get_config(self):
        config = super(DAIN_Layer, self).get_config()
        config.update(
            {
                "mode": self.mode,
                "input_dim": self.input_dim
            }
        )
        return config


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf

    # Create synthetic data (3D time series with 2 features)
    num_samples = 100
    time_steps = 10
    input_dim = 2

    # Generate random data
    data = np.random.randn(250,num_samples, time_steps, input_dim).astype(np.float32)
    print("Input shape:", data.shape)
    # Create a DAIN_Layer instance
    dain_layer = DAIN_Layer(mode='full', input_dim=input_dim)

    intput = tf.keras.layers.Input(np.shape(data))
    output_data = dain_layer(intput)
    model = tf.keras.models.Model(inputs=intput,outputs=output_data)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.MSE
    )
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model_2 = tf.keras.models.load_model(PATH)
    print("Model loaded")
    print(model_2.summary())
    # Check the shape of the output
    print("Input shape:", data.shape)
    print("Output shape:", output_data.shape)
