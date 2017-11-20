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
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.ops import array_ops

import utils

class AutoEncoder(object):
    """AutoEncoder Model"""
    def __init__(self, num_units, cell_type, num_cell_layers, lr=3e-3):
        self.learning_rate = lr
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
        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        # decoder_inputs: [batch_size, max_decoder_time_steps]
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, utils.MAX_CHARS),
                                             name='decoder_inputs')
        # decoder_inputs_length: [batch_size]
        self.decoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

        decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * utils.GO_ID
        decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * utils.EOS_ID

        # decoder_inputs_train: [batch_size , max_time_steps + 1]
        # insert _GO symbol in front of each decoder input
        self.decoder_inputs_train = tf.concat([decoder_start_token,
                                              self.decoder_inputs], axis=1)

        # decoder_inputs_length_train: [batch_size]
        self.decoder_inputs_length_train = self.decoder_inputs_length + 1

        # decoder_targets_train: [batch_size, max_time_steps + 1]
        # insert EOS symbol at the end of each decoder input
        self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                               decoder_end_token], axis=1)

        # decoder_outputs: [batch_size, max_decoder_time_steps]
        # self.decoder_outputs = tf.placeholder(dtype=tf.int32,
        #                                      shape=(None, utils.MAX_CHARS),
        #                                      name='decoder_outputs')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.build_encoder()
        self.build_decoder()

        trainable_params = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
         # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        max_gradient_norm = 1.0
        clip_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        # Update the model
        self.updates = self.optimizer.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def build_encoder(self):
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


    def build_decoder(self):
        print("Building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()
            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[utils.MAX_CHARS, self.embedding_size],
                initializer=initializer, dtype=tf.float32)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.num_units, dtype=tf.float32, name='input_projection', use_bias=False)
            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(utils.MAX_CHARS, name='output_projection', use_bias=False)
            # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.decoder_embeddings, ids=self.decoder_inputs_train)
            # Embedded inputs having gone through input projection layer
            self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)
            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                               sequence_length=self.decoder_inputs_length_train,
                                               time_major=False,
                                               name='training_helper')

            training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)
            # Maximum decoder time_steps in current batch
            max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

            # decoder_outputs_train: BasicDecoderOutput
            #                        namedtuple(rnn_outputs, sample_id)
            # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
            #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
            # decoder_outputs_train.sample_id: [batch_size], tf.int32
            (self.decoder_outputs_train, self.decoder_last_state_train, 
             self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                decoder=training_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=max_decoder_length))

            # More efficient to do the projection on the batch-time-concatenated tensor
            # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
            # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
            self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output) 
            # Use argmax to extract decoder symbols to emit
            self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                name='decoder_pred_train')

            # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
            masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train, 
                                     maxlen=max_decoder_length, dtype=tf.float32, name='masks')
            # Computes per word average cross-entropy over a batch
            # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
            self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train, 
                                              targets=self.decoder_targets_train,
                                              weights=masks,
                                              average_across_timesteps=True,
                                              average_across_batch=True)



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

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.num_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length)

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for i in range(self.num_cell_layers)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            # Essential when use_residual=True
            _input_layer = Dense(self.num_units, dtype=tf.float32,
                                 name='attn_input_feeding', use_bias=False)
            return _input_layer(array_ops.concat([inputs, attention], -1))

         # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.num_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        initial_state = [state for state in encoder_last_state]
        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state
        
    def train(self, sess, encoder_inputs, encoder_inputs_length, 
              decoder_inputs, decoder_inputs_length):
        output_feed = [self.updates,    # Update Op that does optimization
                       self.loss,   # Loss for current batch
                       self.decoder_pred_train] # Decoder prediction
        input_feed = {self.encoder_inputs : encoder_inputs,
                      self.encoder_inputs_length : encoder_inputs_length,
                      self.decoder_inputs : decoder_inputs,
                      self.decoder_inputs_length : decoder_inputs_length
                    }
        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]




