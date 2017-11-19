import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

import utils

class AutoEncoder(object):
    """AutoEncoder Model"""
    def __init__(self, num_units, cell_type, num_cell_layers, lr=3e-3):
        self.lr = lr
        self.num_units = num_units
        self.num_cell_layers = num_cell_layers
        self.cell_type = cell_type
        self.embedding_size = 500
        self.projection_hidden_units = 256
        # encoder_inputs: [batch_size, max_encoder_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, utils.MAX_CHARS),
                                             name='encoder_inputs')
        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,
                                                    shape=(None,),
                                                    name='encoder_inputs_length')

        # decoder_inputs: [batch_size, max_decoder_time_steps]
        # self.decoder_inputs = tf.placeholder(dtype=tf.int32,
        #                                      shape=(None, utils.MAX_CHARS),
        #                                      name='decoder_inputs')

        # decoder_outputs: [batch_size, max_decoder_time_steps]
        # self.decoder_outputs = tf.placeholder(dtype=tf.int32,
        #                                      shape=(None, utils.MAX_CHARS),
        #                                      name='decoder_outputs')

        print("Building encoder..")
        with tf.variable_scope('encoder'):
            # Build RNN cell
            self.encoder_cell = self.build_encoder_cell()
            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[utils.VOCAB_SIZE,
                                                             self.embedding_size],
                                                      initializer=initializer,
                                                      dtype=tf.float32)
            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                                                params=self.encoder_embeddings,
                                                ids=self.encoder_inputs)
            
            input_layer = Dense(self.projection_hidden_units, use_bias=False,
                dtype=tf.float32, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=tf.float32,
                time_major=False)

    def build_single_cell(self):
        
        if self.cell_type == 'gru':
            cell_class = GRUCell
        else:
            cell_class = LSTMCell
        cell = cell_class(self.num_units)

        # if self.use_dropout:
        #     cell = DropoutWrapper(cell, dtype=self.dtype,
        #                           output_keep_prob=self.keep_prob_placeholder,)
        # if self.use_residual:
        #     cell = ResidualWrapper(cell)
            
        return cell

    # Building encoder cell
    def build_encoder_cell(self):
        return MultiRNNCell([self.build_single_cell() for i in range(self.num_cell_layers)])
            



