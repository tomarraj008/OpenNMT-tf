"""RNN cells helpers."""

import collections

import tensorflow as tf


class _RNNCellWrapper(tf.keras.layers.Wrapper):
  """Helper class to wrap a RNN cell."""

  @property
  def state_size(self):
    """The cell state size."""
    return self.layer.state_size

  @property
  def output_size(self):
    """The cell output size."""
    return self.layer.output_size

  def build(self, input_shape):
    """Build the cell."""
    self.layer.build(input_shape)
    self.built = True

  def call(self, inputs, states, training=None):
    """Calls the cell."""
    return self.layer.call(inputs, states, training=training)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Returns the cell initial state."""
    return self.layer.get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)


class ResidualWrapper(_RNNCellWrapper):
  """Adds residual connection to a RNN cell."""

  def call(self, inputs, states, training=None):
    outputs, states = super(ResidualWrapper, self).call(
        inputs, states, training=training)
    return outputs + inputs, states


class DropoutWrapper(_RNNCellWrapper):
  """Applies dropout to the input and output of a RNN cell."""

  def __init__(self, layer, input_rate=0.0, output_rate=0.0, **kwargs):
    super(DropoutWrapper, self).__init__(layer, **kwargs)
    self.input_rate = input_rate
    self.output_rate = output_rate

  def call(self, inputs, states, training=None):
    if training and self.input_rate > 0:
      inputs = tf.nn.dropout(inputs, rate=self.input_rate)
    outputs, states = super(DropoutWrapper, self).call(
        inputs, states, training=training)
    if training and self.output_rate > 0:
      outputs = tf.nn.dropout(outputs, rate=self.output_rate)
    return outputs, states

  def get_config(self):
    """Returns the config if this cell wrapper."""
    config = super(DropoutWrapper, self).get_config()
    config["input_rate"] = self.input_rate
    config["output_rate"] = self.output_rate
    return config


_CUSTOM_CELLS = {
    "DropoutWrapper": DropoutWrapper,
    "ResidualWrapper": ResidualWrapper,
}


def build_keras_cell(num_layers,
                     num_units,
                     dropout=0.0,
                     residual_connections=False,
                     cell_class=tf.keras.layers.LSTMCell):
  """Builds a multi-layer Keras RNN cell.

  Args:
    num_layers: The number of layers.
    num_units: The number of units in each layer.
    dropout: The probability to drop units in each layer output.
    residual_connections: If ``True``, each layer input will be added to its output.
    cell_class: The inner cell class or a callable taking :obj:`num_units` as
      argument and returning a cell.

  Returns:
    A ``tf.keras.layers.Layer`` that acts as a RNN cell.
  """
  cells = []
  for l in range(num_layers):
    cell = cell_class(num_units)
    if dropout > 0:
      cell = DropoutWrapper(cell, output_rate=dropout)
    if residual_connections and l > 0:
      cell = ResidualWrapper(cell)
    cells.append(cell)
  return tf.keras.layers.StackedRNNCells(cells)

def build_rnn(num_layers,
              num_units,
              bidirectional=False,
              dropout=0.0,
              residual_connections=False,
              cell_class=tf.keras.layers.LSTMCell):
  """Builds a RNN layer.

  Args:
    num_layers: The number of layers.
    num_units: The number of units in each layer.
    bidirectional: If ``True``, build a bidirectional RNN.
    dropout: The probability to drop units in each layer output.
    residual_connections: If ``True``, each layer input will be added to its output.
    cell_class: The inner cell class or a callable taking :obj:`num_units` as
      argument and returning a cell.

  Returns:
    A ``tf.keras.layers.Layer``.
  """
  cell = build_keras_cell(
      num_layers,
      num_units,
      dropout=dropout,
      residual_connections=residual_connections,
      cell_class=cell_class)
  with tf.keras.utils.custom_object_scope(_CUSTOM_CELLS):
    layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    if bidirectional:
      layer = tf.keras.layers.Bidirectional(layer, merge_mode=None)
  return layer

def build_cell(num_layers,
               num_units,
               dropout=0.0,
               residual_connections=False,
               cell_class=tf.nn.rnn_cell.LSTMCell,
               attention_layers=None,
               attention_mechanisms=None):
  """Convenience function to build a multi-layer RNN cell.

  Args:
    num_layers: The number of layers.
    num_units: The number of units in each layer.
    dropout: The probability to drop units in each layer output.
    residual_connections: If ``True``, each layer input will be added to its output.
    cell_class: The inner cell class or a callable taking :obj:`num_units` as
      argument and returning a cell.
    attention_layers: A list of integers, the layers after which to add attention.
    attention_mechanisms: A list of ``tf.contrib.seq2seq.AttentionMechanism``
      with the same length as :obj:`attention_layers`.

  Returns:
    A ``tf.nn.rnn_cell.RNNCell``.

  Raises:
    ValueError: if :obj:`attention_layers` and :obj:`attention_mechanisms` do
      not have the same length.
  """
  cells = []

  attention_mechanisms = attention_mechanisms or []
  attention_layers = attention_layers or []

  if len(attention_mechanisms) != len(attention_layers):
    raise ValueError("There must be the same number of attention mechanisms "
                     "as the number of attention layers")

  for l in range(num_layers):
    cell = cell_class(num_units)
    if l in attention_layers:
      cell = tf.contrib.seq2seq.AttentionWrapper(
          cell,
          attention_mechanisms[attention_layers.index(l)],
          attention_layer_size=num_units)
    if dropout > 0.0:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
    if residual_connections and l > 0:
      cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    cells.append(cell)

  if len(cells) == 1:
    return cells[0]
  else:
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def last_encoding_from_state(state):
  """Returns the last encoding vector from the state.

  For example, this is the last hidden states of the last LSTM layer for a
  LSTM-based encoder.

  Args:
    state: The encoder state.

  Returns:
    The last encoding vector.
  """
  if isinstance(state, collections.Sequence):
    state = state[-1]
  if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
    return state.h
  return state
