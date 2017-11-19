import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from model import AutoEncoder as Model
import utils

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
FLAGS = tf.app.flags.FLAGS

def main(*args):
    # Hyper parameters
    training_steps = 1000000
    valid_step = 100
    num_encoder_units = 512

    dataset = utils.DataSet('data/head_cleaned.smi')
    model = Model(num_encoder_units, 'gru', 3)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        training_batch, sequence_lens = dataset.get_batch(FLAGS.batch_size, 'train')
        encoder_outputs = sess.run(model.encoder_outputs,feed_dict={
                        model.encoder_inputs:training_batch,
                        model.encoder_inputs_length: sequence_lens
                })
        print(encoder_outputs)
        # print(training_batch)
        # for step in range(training_steps):
        #     if (step % valid_step) == 0:
        #         pass

        # embedding = sess.run(model.encoder_embeddings, feed_dict={})
        # print(embedding)
        # embedding_lookup = sess.run(model.encoder_inputs_embedded, feed_dict={model.encoder_inputs:training_batch})
        # print(embedding_lookup)

if __name__ == "__main__":
    tf.app.run()
