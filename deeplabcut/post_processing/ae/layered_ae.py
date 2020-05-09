import numpy as np
import os
import pandas as pd
import pdb
import random
import copy
import math
from scipy import signal
import tensorflow as tf
import sys
import time
import data_input

class VAE(object):
  def __init__(self,network_arch, batch_size, train_epochs, file, model_path, model_summary="model_summary", saved_model=False):
    tf.reset_default_graph()
    encode1, encode2, encode3, decode0, decode1, decode2, latent_size, input_size, output_size = [network_arch[i] for i in network_arch]

    self.batch_size = batch_size
    self.train_epochs = train_epochs
    self.file = file
    self.input_size = input_size
    self.intermediate_size = output_size * 3
    self.output_size = output_size
    self.time_pair = int(input_size / output_size)
    self.model_path = model_path

    # Data input pipeline
    #inputData = (train_data_noisy, train_data_denoised)
    self.training_x = tf.placeholder(tf.float32, shape=[None, self.input_size])
    self.training_y = tf.placeholder(tf.float32, shape=[None, self.output_size])
    self.training_z = tf.placeholder(tf.float32, shape=[None, self.intermediate_size])
    trainingData = tf.data.Dataset.from_tensor_slices((self.training_x, self.training_y, self.training_z))
    trainingData = trainingData.shuffle(batch_size*5).batch(batch_size).prefetch(1)
    trainingData = trainingData.cache()
    trainingData = trainingData.repeat(train_epochs)

    self.predict_x = tf.placeholder(tf.float32, shape=[None, self.input_size])
    predictingData = tf.data.Dataset.from_tensor_slices(self.predict_x)
    predictingData = predictingData.batch(batch_size)
    predictingData = predictingData.map(lambda x: data_input.map_pred3(x, self.output_size, self.batch_size))

    iterator = tf.data.Iterator.from_structure(trainingData.output_types,
                                               trainingData.output_shapes)

    transfer_fct = tf.nn.leaky_relu
    self.learning_rate = 0.001
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.training_init = iterator.make_initializer(trainingData)
    self.predicting_init = iterator.make_initializer(predictingData)

    # Input and output for model
    x_in, x_out, x_inter = iterator.get_next()

    # Use He Initialization - Best for Leaky ReLu
    init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                         mode='FAN_IN',
                                                         uniform=False)
    # Encoder
    layer_1 = tf.contrib.layers.fully_connected(x_in, encode1, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())
    layer_2 = tf.contrib.layers.fully_connected(layer_1, encode2, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())
    layer_3 = tf.contrib.layers.fully_connected(layer_2, encode3, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())

    self.z_mean = tf.contrib.layers.fully_connected(layer_3, latent_size, activation_fn=transfer_fct,
                                         weights_initializer=init,biases_initializer=tf.zeros_initializer())

    # Decoder
    dlayer_0 = tf.contrib.layers.fully_connected(self.z_mean, decode0, activation_fn=transfer_fct,
                                      weights_initializer=init,biases_initializer=tf.zeros_initializer())
    dlayer_1 = tf.contrib.layers.fully_connected(dlayer_0, decode1, activation_fn=transfer_fct,
                                      weights_initializer=init,biases_initializer=tf.zeros_initializer())
    dlayer_2 = tf.contrib.layers.fully_connected(dlayer_1, decode2, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())

    self.x_residuals = tf.contrib.layers.fully_connected(dlayer_2, self.intermediate_size, activation_fn=None,
                                                 weights_initializer=init,biases_initializer=tf.zeros_initializer())

    self.x_reconstr_mean_one = tf.add(self.x_residuals,x_in[:,self.output_size:self.output_size*(int(self.time_pair/2)+2)])
    self.x_reconstr_mean_one = transfer_fct(self.x_reconstr_mean_one)
    self.x_reconstr_normalized = tf.layers.batch_normalization(self.x_reconstr_mean_one, axis=1, training=self.is_training)

    # Encoder
    layer_4 = tf.contrib.layers.fully_connected(self.x_reconstr_normalized, encode1, activation_fn=transfer_fct,
                                                weights_initializer=init,biases_initializer=tf.zeros_initializer())
    layer_5 = tf.contrib.layers.fully_connected(layer_4, encode2, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())
    layer_6 = tf.contrib.layers.fully_connected(layer_5, encode3, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())
    self.z_mean_two = tf.contrib.layers.fully_connected(layer_6, latent_size, activation_fn=transfer_fct,
                                         weights_initializer=init,biases_initializer=tf.zeros_initializer())

    dlayer_3 = tf.contrib.layers.fully_connected(self.z_mean_two, decode0, activation_fn=transfer_fct,
                                      weights_initializer=init,biases_initializer=tf.zeros_initializer())
    dlayer_4 = tf.contrib.layers.fully_connected(dlayer_3, decode1, activation_fn=transfer_fct,
                                      weights_initializer=init,biases_initializer=tf.zeros_initializer())
    dlayer_5 = tf.contrib.layers.fully_connected(dlayer_4, decode2, activation_fn=transfer_fct,
                                     weights_initializer=init,biases_initializer=tf.zeros_initializer())

    self.x_residuals_two = tf.contrib.layers.fully_connected(dlayer_5, output_size, activation_fn=None,
                                                 weights_initializer=init,biases_initializer=tf.zeros_initializer())
    self.x_reconstr_mean_two = tf.add(self.x_residuals_two, self.x_reconstr_mean_one[:,self.output_size:self.output_size*2])



    # squared error
    reconstr_loss_one = tf.reduce_sum(tf.square(self.x_reconstr_mean_one - x_inter),1)
    reconstr_loss_two = tf.reduce_sum(tf.square(self.x_reconstr_mean_two - x_out),1)

    #  The latent loss, which is defined as the Kullback Leibler divergence
    ##    between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #  https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

    # latent_loss = 0.5 * tf.reduce_sum(-1 - z_log_sigma_sq
    #                                   + tf.square(self.z_mean)
    #                                    + tf.exp(z_log_sigma_sq), 1)
    self.proportion_intermediate = tf.placeholder(tf.float32, shape=())
    self.proportion_end = tf.placeholder(tf.float32, shape=())
    self.cost = self.proportion_intermediate*tf.reduce_mean(reconstr_loss_one) + self.proportion_end*tf.reduce_mean(reconstr_loss_two)   # average over batch
    tf.summary.scalar('cost', self.cost)

    # Use ADAM optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    # Merge all the summaries and write them out to /tmp/train/1
    self.merged = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())
    self.train_writer = tf.summary.FileWriter("./training_output/"+model_summary)
    # train_writer.add_graph(sess.graph)
    if saved_model:
      # Launch the session
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.model_path)
    else:
      # Initializing the tensor flow variables
      init = tf.global_variables_initializer()
      self.sess = tf.Session()
      self.sess.run(init)


  def train(self,datax,datainter, datay, inter_prop=0.5, end_prop=0.5, move_iter=100000, save_iter=50000, print_iter=500):
    start = time.time()
    i = 0
    self.sess.run(self.training_init, feed_dict={self.training_x: datax, self.training_z: datainter, self.training_y: datay})
    #self.file.write("Training Model...\n")
    avgc = 0
    while True:
      try:
        opt, currC, summary = self.sess.run((self.optimizer, self.cost, self.merged),
                                          feed_dict={self.proportion_intermediate: inter_prop,
                                                     self.proportion_end: end_prop,
                                                     self.is_training: True})
        i = i + 1
        avgc = avgc + currC
        if i % print_iter == 0:
            self.train_writer.add_summary(summary,i)
            self.file.write(str(avgc/1000)+"\n")
            avgc = 0
        if i == move_iter:
          inter_prop = 0.0
          end_prop = 1.0
        if i % save_iter == 0:
          self.saver.save(self.sess, self.model_path)
      except tf.errors.OutOfRangeError:
        break
    end = time.time()
    self.file.write("Time to train: "+str(end-start)+"\n")
    self.saver.save(self.sess, self.model_path)

  def reconstruct(self,x):
    self.sess.run(self.predicting_init, feed_dict={self.predict_x: x})
    nparray = np.empty([0,self.output_size])
    while True:
      try:
        nparray = np.append(nparray, self.sess.run(self.x_reconstr_mean_two, feed_dict={self.is_training: False}), axis=0)
      except tf.errors.OutOfRangeError:
        break
    return nparray

  def end(self):
    self.sess.close()
