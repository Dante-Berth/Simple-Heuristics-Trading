import tensorflow as tf


class tsmixer_cnn_dense_model(tf.keras.layers.Layer):
  """CNN model."""

  def __init__(self, n_channel, pred_len, kernel_size,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.cnn = tf.keras.layers.Conv1D(
        n_channel, kernel_size, padding='same'
    )
    self.dense = tf.keras.layers.Dense(pred_len)

  def call(self, x):
    # x: [Batch, Input length, Channel]
    x = self.cnn(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = self.dense(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    return x  # [Batch, Output length, Chann

class tsmixer_fully_linear_model(tf.keras.layers.Layer):
  """Fully linear model."""

  def __init__(self, n_channel, pred_len,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(pred_len * n_channel)
    self.reshape = tf.keras.layers.Reshape((pred_len, n_channel))

  def call(self, x):
    # x: [Batch, Input length, Channel]
    x = self.flatten(x)
    x = self.dense(x)
    x = self.reshape(x)
    return x

class RevNorm(tf.keras.layers.Layer):
  """Reversible Instance Normalization."""

  def __init__(self, axis, eps=1e-5, affine=True,*args,**kwargs):
    super(RevNorm).__init__(*args,**kwargs)
    self.axis = axis
    self.eps = eps
    self.affine = affine

  def build(self, input_shape):
    if self.affine:
      self.affine_weight = self.add_weight(
          'affine_weight', shape=input_shape[-1], initializer='ones'
      )
      self.affine_bias = self.add_weight(
          'affine_bias', shape=input_shape[-1], initializer='zeros'
      )

  def call(self, x, mode, target_slice=None):
    if mode == 'norm':
      self._get_statistics(x)
      x = self._normalize(x)
    elif mode == 'denorm':
      x = self._denormalize(x, target_slice)
    else:
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    self.mean = tf.stop_gradient(
        tf.reduce_mean(x, axis=self.axis, keepdims=True)
    )
    self.stdev = tf.stop_gradient(
        tf.sqrt(
            tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
        )
    )

  def _normalize(self, x):
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    if self.affine:
      x = x - self.affine_bias[target_slice]
      x = x / self.affine_weight[target_slice]
    x = x * self.stdev[:, :, target_slice]
    x = x + self.mean[:, :, target_slice]
    return x

def res_block(inputs, norm_type, activation, dropout, ff_dim):
  # Mixer Layer
  """Residual block of TSMixer."""

  norm = (
      tf.keras.layers.LayerNormalization
      if norm_type == 'L'
      else tf.keras.layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = tf.keras.layers.Dense(x.shape[-1], activation=activation)(x)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
  x = tf.keras.layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  x = norm(axis=[-2, -1])(res)
  x = tf.keras.layers.Dense(ff_dim, activation=activation)(
      x
  )  # [Batch, Input Length, FF_Dim]
  x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
  x = tf.keras.layers.Dropout(dropout)(x)
  return x + res

@tf.keras.utils.register_keras_serializable(name="tsmixer_res_block")
class tsmixer_res_block(tf.keras.layers.Layer):
  def __init__(self,norm_type, activation, dropout, ff_dim,**kwargs):
    super(tsmixer_res_block,self).__init__(**kwargs)
    self.norm_type = norm_type
    self.activation = activation
    self.dropout = dropout
    self.ff_dim = ff_dim
    self.norm = (
            tf.keras.layers.LayerNormalization
            if norm_type == 'L'
            else tf.keras.layers.BatchNormalization
        )
    self.dense_temporal_projection = None
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.dense_time_mixing = None
    self.dense_feature_mixing = None

  def get_config(self):
    config = super(tsmixer_res_block, self).get_config()
    config.update({
      'norm_type': self.norm_type,
      'activation': self.activation,
      'dropout': self.dropout,
      'ff_dim': self.ff_dim
    })
    return config
  def build(self,input_shape,*args,**kwargs):
    self.dense_temporal_projection = tf.keras.layers.Dense(input_shape[-2], activation=self.activation)
    self.dense_feature_mixing = tf.keras.layers.Dense(input_shape[-1], activation=self.activation)
    self.dense_time_mixing = tf.keras.layers.Dense(self.ff_dim, activation=self.activation)
  def call(self, inputs,*args,**kwargs):

    x = self.norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = self.dense_temporal_projection(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = self.dropout(x)
    res = x + inputs

    # Feature Linear
    x = self.norm(axis=[-2, -1])(res)
    x = self.dense_time_mixing(x)  # [Batch, Input Length, FF_Dim]
    x = self.dropout(x)
    x = self.dense_feature_mixing(x)  # [Batch, Input Length, Channel]
    x = self.dropout(x)
    return x + res

def build_model(
      input_shape,
      pred_len,
      norm_type,
      activation,
      n_block,
      dropout,
      ff_dim,
      target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
      x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
      x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = tf.keras.layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

  return tf.keras.Model(inputs, outputs)
if __name__ == "__main__":
  x = tf.ones((12,24,36))
  inputs =  tf.keras.layers.Input(shape=(24, 36))
  tsmixer = tsmixer_res_block(norm_type="L", activation="gelu", dropout=0.3, ff_dim=15)
  print(tsmixer(x).shape)
  y = tsmixer(inputs)

  model = tf.keras.models.Model(inputs=inputs, outputs=y)
  model.compile(
    optimizer="Adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
  )

  print("Model before the loading")
  model.summary()

  PATH = 'testing_model_custom_activation_layer.h5'
  model.save(PATH)
  model = tf.keras.models.load_model(PATH)
  print("Model loaded")
  print(model.summary())