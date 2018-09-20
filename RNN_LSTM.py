# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
GPUconfig = tf.ConfigProto()
GPUconfig.gpu_options.per_process_gpu_memory_fraction = 0.2
import numpy as np
import time
import sys
import pickle
import argparse
import copy
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class Config(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.8
  learning_rate = 1.0
  init_scale = 0.05
  num_epochs = 50
  max_epoch = 6
  word_vocab_size = 0 # to be determined later
  num_layers = 2

  # RNN hyperparameters
  num_steps = 35
  hidden_size = 650

  keep_prob =0.5

def read_data(config):
  '''read data sets, construct all needed structures and update the config'''
  word_data = open('data/word/train.txt', 'r').read().replace('\n', '<eos>').split()
  words = list(set(word_data))

  word_data_size, word_vocab_size = len(word_data), len(words)
  print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
  config.word_vocab_size = word_vocab_size

  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', '<eos>').split()
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data('data/word/train.txt')
  valid_raw_data = get_word_raw_data('data/word/valid.txt')
  test_raw_data = get_word_raw_data('data/word/test.txt')

  return train_raw_data, valid_raw_data, test_raw_data

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class LSTM_RNNCell(tf.nn.rnn_cell.RNNCell):
  
  def __init__(self, hidden_size, reuse=True):
    #self._num_units = num_units
    #self._activation = activation or math_ops.tanh
    self._hidden_size = hidden_size
    self.init_scale = init_scale = config.init_scale = 0.05

  @property
  def state_size(self):
    return LSTMStateTuple(self._hidden_size, self._hidden_size)

  @property
  def output_size(self):
    return self._hidden_size

  def __call__(self, inputs, state, scope=None):
    """LSTM Zaremba network:
    where Wx = """
    c, h = state
    #print(state.shape)
    with tf.variable_scope(scope or "LSTM"):
        
        with tf.variable_scope("Gates"):
            W_i = tf.get_variable('W_i', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            U_i = tf.get_variable('U_i', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            b_i = tf.get_variable('b_i', [self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(h, U_i) + b_i)
            W_f = tf.get_variable('W_f', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            U_f = tf.get_variable('U_f', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            b_f = tf.get_variable('b_f', [self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(h, U_f) + b_f)
            W_o = tf.get_variable('W_o', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            U_o = tf.get_variable('U_o', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            b_o = tf.get_variable('b_o', [self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            #o = tf.sigmoid(tf.matmul(state, U_o) + b_o)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(h, U_o) + b_o)
        with tf.variable_scope("Candidate"):
            W_g = tf.get_variable('W_g', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            U_g = tf.get_variable('U_g', [self._hidden_size, self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            b_g = tf.get_variable('b_g', [self._hidden_size], initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
            #g = f*c + i*tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(state, U_g) + b_g)
            g = tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(h, U_g) + b_g)
        
        new_c = (c * f + i * g)
        new_h = tf.tanh(new_c) * o
        
        new_state = LSTMStateTuple(new_c, new_h)
        #output = tf.softmax(W_y*mt + b_y)
    return new_h, new_state

#_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class batch_producer(object):
  '''Slice the raw data into batches'''
  def __init__(self, raw_data, batch_size, num_steps):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    self.batch_len = len(self.raw_data) // self.batch_size
    self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                           (self.batch_size, self.batch_len))
    
    self.epoch_size = (self.batch_len - 1) // self.num_steps
    self.i = 0
  
  def __next__(self):
    if self.i < self.epoch_size:
      # batch_x and batch_y are of shape [batch_size, num_steps]
      batch_x = self.data[::, 
          self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
      batch_y = self.data[::, 
          self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
      self.i += 1
      return (batch_x, batch_y)
    else:
      raise StopIteration()

  def __iter__(self):
    return self

class MyModel:
  def __init__(self, config, is_train):
    # get hyperparameters
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    init_scale = config.init_scale
    word_emb_dim = hidden_size = config.hidden_size
    word_vocab_size = config.word_vocab_size
    
    #initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    
    
    # language model 
  #with tf.variable_scope('model', initializer=initializer):
    # embedding matrix
    word_embedding = tf.get_variable("word_embedding", [word_vocab_size, word_emb_dim])

    # placeholders for training data and labels
    self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
    self.y = tf.placeholder(tf.int32, [batch_size, num_steps])

    # we first embed words ...
    words_embedded = tf.nn.embedding_lookup(word_embedding, self.x)
    if is_train:
      rnn_input = tf.nn.dropout(words_embedded, config.keep_prob)#dropOut = words_embedd
    # ... and then process it with a stack of two LSTMs
    if is_train:
      rnn_input = tf.unstack(rnn_input, axis=1)
    else:
      rnn_input = tf.unstack(words_embedded, axis=1)

    # basic RNN cell
    cell1 = LSTM_RNNCell(hidden_size)
    if is_train:
      cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=config.keep_prob)
    cell2 = LSTM_RNNCell(hidden_size)
    if is_train:
      cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    
    self.init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    state = self.init_state

    outputs, self.state = tf.contrib.rnn.static_rnn(
        cell, 
        rnn_input, 
        dtype=tf.float32, 
        initial_state=self.init_state)
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

    # softmax layer    
    weights = tf.get_variable('weights', [hidden_size, word_vocab_size], dtype=tf.float32)
    biases = tf.get_variable('biases', [word_vocab_size], dtype=tf.float32)

    logits = tf.matmul(output, weights) + biases
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self.y, [-1])],
        [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self.cost = cost = tf.reduce_sum(loss) / batch_size
    
    # training
    self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), 
                                     global_step = tf.contrib.framework.get_or_create_global_step())
    
    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)
    
    self.final_state =state
    
  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

#session.run(lr_update, feed_dict={new_lr: lr_value})


# In[ ]:


def model_size():
  '''finds the total number of trainable variables a.k.a. model size'''
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


def run_epoch(sess, m_train, raw_data, train_op, config, is_train=False, lr=None):
  start_time = time.time()
  if is_train: m_train.assign_lr(sess, lr)
  #if is_train: sess.run(lr_update, feed_dict={new_lr: lr})

  iters = 0
  costs = 0
  state_val = sess.run(m_train.init_state)

  batches = batch_producer(raw_data, config.batch_size, config.num_steps)

  for (batch_x, batch_y) in batches:
    # run the model on current batch
    #if is_train:
    _, c, state_val = sess.run(
        [train_op, m_train.cost, m_train.state],
        feed_dict={m_train.x: batch_x, m_train.y: batch_y, 
                   m_train.init_state: state_val})
    #else:
     # c, state_val = sess.run([cost, state], 
      #    feed_dict={x: batch_x, y: batch_y, 
       #              init_state: state_val})

    costs += c
    step = iters // config.num_steps
    if is_train and step % (batches.epoch_size // 10) == 10:
      print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
      print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
      print('speed =', 
          round(iters * config.batch_size / (time.time() - start_time)), 
          'wps')
    iters += config.num_steps
  
  return np.exp(costs / iters)


# In[ ]:


if __name__ == '__main__':
    
  config = Config();
  train_raw_data, valid_raw_data, test_raw_data = read_data(config)
    
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

  with tf.variable_scope('Model', reuse=False, initializer=initializer):
    m_train = MyModel(config=config, is_train=True)
    
  print('Model size is: ', model_size())

  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    m_valid = MyModel(config=config, is_train=False)
    
  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    m_test = MyModel(config=config, is_train=False)

  saver = tf.train.Saver()
    
  num_epochs = config.num_epochs
  init = tf.global_variables_initializer()
  learning_rate = config.learning_rate

  with tf.Session(config=GPUconfig) as sess:
    sess.run(init)
    prev_valid_ppl = float('inf')
    best_valid_ppl = float('inf')

    for epoch in range(num_epochs):
      train_ppl = run_epoch(
          sess, m_train, train_raw_data, m_train.train_op, config, is_train=True, 
          lr=learning_rate)
      print('epoch', epoch + 1, end = ': ')
      print('train ppl = %.3f' % train_ppl, end=', ')
      print('lr = %.3f' % learning_rate, end=', ')

      # Get validation set perplexity
      valid_ppl = run_epoch(
          sess, m_valid, valid_raw_data, tf.no_op(), config, is_train=False)
      print('valid ppl = %.3f' % valid_ppl)
        
      # Update the learning rate if necessary
      if epoch + 2 > config.max_epoch: learning_rate *= config.lr_decay
        
      # Save model if it gives better valid ppl
      if valid_ppl < best_valid_ppl:
        save_path = saver.save(sess, 'saves/model.ckpt')
        print('Valid ppl improved. Model saved in file: %s' % save_path)
        best_valid_ppl = valid_ppl

  '''Evaluation of a trained model on test set'''
  with tf.Session(config=GPUconfig) as sess:
    # Restore variables from disk.
    saver.restore(sess, 'saves/model.ckpt')
    print('Model restored.')

    # Get test set perplexity
    test_ppl = run_epoch(sess, m_test, test_raw_data, tf.no_op(), config, is_train=False)
print('Test set perplexity = %.3f' % test_ppl)
