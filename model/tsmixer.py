import tensorflow as tf


class tsmixer_cnn_dense_model(tf.keras.layers.Layer):
  """CNN model."""

  def __init__(self, n_channel, pred_len, kernel_size,*args,**kwargs):
    super(tsmixer_cnn_dense_model).__init__(*args,**kwargs)
    self.cnn = tf.keras.layers.Conv1D(
        n_channel, kernel_size, padding='same', input_shape=(None, n_channel)
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
  """Residual block of TSMixer."""

  norm = (
      tf.keras.layers.LayerNormalization
      if norm_type == 'L'
      else tf.keras.layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x =  tf.keras.layers.Dense(x.shape[-1], activation=activation)(x)
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