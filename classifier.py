# -*- coding: utf-8 -*-
"""
@author: CK
"""

import os
import sys
import numpy as np
import time
import math
import cPickle
import cv2
import argparse
import logging
from datetime import datetime
import scipy.ndimage
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.platform
from glob import glob
import random
import threading


def conv_layer(input_fmaps, conv_parameters, kh, kw, n_out_fmaps, name, y_stride=1, x_stride=1):
    """ convolution layer
    Args:
    input_fmaps: input feature-maps
    kw: kernel width
    kh: kernel height
    y_stride: stride length along height
    x_stride: stride length along width
    name: layer name

    Returns:
    out: Output f-maps
    """
    n_in_fmaps = input_fmaps.get_shape()[-1].value
    init_range = math.sqrt(
        6.0 / (kh * kw * n_in_fmaps + kh * kw * n_out_fmaps))

    with tf.name_scope(name) as scope:
        kernel_init_val = tf.random_uniform(
            [kh, kw, n_in_fmaps, n_out_fmaps], dtype=tf.float32, minval=-init_range, maxval=init_range)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        bias_init_val = tf.constant(0.0, shape=[n_out_fmaps], dtype=tf.float32)
        bias = tf.Variable(bias_init_val, trainable=True, name='b')
        out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_fmaps, kernel, strides=[
                         1, y_stride, x_stride, 1], padding='SAME'), bias))
        conv_parameters += [kernel, bias]

    return out


def max_pool(input_fmaps, name, y_stride=2, x_stride=2, kw=2, kh=2):
    """max pooling layer

    Args:
    input_fmaps: input feature maps
    y_stride: stride length along height
    x_stride: stride length along width
    kw: kernel width
    kh: kernel height
    name: layer name

    Returns:
    out: f-maps after applying max pooling

    """

    with tf.name_scope(name) as scope:
        out = tf.nn.max_pool(input_fmaps, ksize=[1, kh, kw, 1], strides=[
                             1, y_stride, x_stride, 1], padding='SAME')

    return out



def fc_layer(input_fmaps, conv_parameters, n_out, name):
    """Fully connected layer

    Args:
    input_fmaps: input feature-maps
    n_out: number of neurons in fc layer
    w: weight matrix
    b = bias
    name: name of the layer

    Returns:
    out: activation of the fully connected layer
    """
    n_in = input_fmaps.get_shape()[-1].value
    init_range = math.sqrt(6.0 / (n_in + n_out))

    with tf.name_scope(name) as scope:
        kernel_init_val = tf.random_uniform(
            [n_in, n_out], dtype=tf.float32, minval=-init_range, maxval=init_range)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        bias = tf.Variable(bias_init_val, trainable=True, name='b')
        out = tf.add(tf.matmul(input_fmaps, kernel), bias)

        conv_parameters +=[kernel, bias]

    return out


def conv_model(img, conv_parameters, n_classes):
    """Convolution neural net model (VGG model)
    Args:
    img: input image
    n_classes: number of output classes
    conv_parameters: list of all the convolution layers parameters (weights and biases)

    Returns:
    out: output of the final fully connected layer (apply softmax to it)
    """
    conv1_1 = conv_layer(img, conv_parameters, kh=3, kw=3,
                         n_out_fmaps=64, name='conv1_1')
    conv1_2 = conv_layer(conv1_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=64, name='conv1_2')
    pool1 = max_pool(conv1_2, name='pool1')

    conv2_1 = conv_layer(pool1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=128, name='conv2_1')
    conv2_2 = conv_layer(conv2_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=128, name='conv2_2')
    pool2 = max_pool(conv2_2, name='pool2')

    conv3_1 = conv_layer(pool2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_1')
    conv3_2 = conv_layer(conv3_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_2')
    conv3_3 = conv_layer(conv3_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_3')
    pool3 = max_pool(conv3_3, name='pool3')

    conv4_1 = conv_layer(pool3, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_1')
    conv4_2 = conv_layer(conv4_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_2')
    conv4_3 = conv_layer(conv4_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_3')
    pool4 = max_pool(conv4_3, name='pool4')

    conv5_1 = conv_layer(pool4, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_1')
    conv5_2 = conv_layer(conv5_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_2')
    conv5_3 = conv_layer(conv5_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_3')
    pool5 = max_pool(conv5_3, name='pool5')
    
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    reshaped_layer = tf.reshape(pool5, [-1, flattened_shape])

    fc6 = fc_layer(reshaped_layer, conv_parameters, n_out=4096, name='fc6')
    relu6 = tf.nn.relu(fc6)

    fc7 = fc_layer(relu6, conv_parameters, n_out=4096, name='fc7')
    relu7 = tf.nn.relu(fc7)

    fc8 = fc_layer(relu7, conv_parameters, n_out=n_classes, name='fc_softmax')

    # # convoution layer with ReLU and GAP
    # conv6_1 = conv_layer(conv5_3, conv_parameters, kh=3,
    #                      kw=3, n_out_fmaps=n_gap_channels, name='conv6_1')
    # gap = avg_pool(conv6_1, name='gap', y_stride=1, x_stride=1, kh=14, kw=14)

    # # flatten the data to pass it to the softmax layer
    # shp = gap.get_shape()
    # flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # reshaped_gap = tf.reshape(gap, [-1, flattened_shape])

    # # final output layer
    # out = fc_layer(reshaped_gap, n_out=n_classes, name='fc_softmax')

    return fc8


def loss_function(fc_softmax, y):
    """loss function

    Args:
    fc_softmax: the final layer of the convolution model
    y: ground truth labels

    Returns:
    cost: tensorflow loss variable

    """
    # generate one hot vectors for the ground truth labels
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, y])
    one_hot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, n_classes]), 1.0, 0.0)

    # apply softmax function on the flattened global average pooled layer and
    # calcualte loss using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        fc_softmax, one_hot_labels)
    cost = tf.reduce_mean(cross_entropy)

    return cost



def initialize_conv_layers(weights_file_path, conv_parameters=None, sess=None):
    """use the weights of pre trained 16 layer VGG model on imagenet
    dataset to initialize the layers of the current model.
    Weights are downloaded from: http://www.cs.toronto.edu/~frossard/post/vgg16/

    Args:
    weights_file_path: path of the  pre trained vgg model
    """
    weights = np.load(weights_file_path)
    keys = sorted(weights.keys())
    # number of layers in the VGG model used (total 16 layers)
    # (convolution layer weights and biases for 13 cnn layers and 2 fc layers)
    n_layers = 15*2 # (kernel and bias for each layer)

    if sess and conv_parameters:
        for i, k in enumerate(keys[:n_layers]):
            logging.info("{}, {}, {}".format(i, k, np.shape(weights[k])))
            sess.run(conv_parameters[i].assign(weights[k]))


def train():
    """ Intializes the convolution model with global average pooling and trains it on the
    input data

    """
    # generate the tensorflow computation graph
    with tf.Graph().as_default():
        # tf graph input
        x = tf.placeholder(
            'float', [None, img_height, img_width, n_img_channels], name="InputData")
        y = tf.placeholder(tf.int32, [None], name="Labels")

        # contruct CNN model
        conv_parameters = []  # list of cnn layers parameters
        cnn_out = conv_model(x, conv_parameters, n_classes=n_classes)

        # define loss and optimizer
        cost = loss_function(cnn_out, y)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)

        # evaluate model
        correct_pred = tf.equal(tf.cast(tf.argmax(cnn_out, 1), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # create training summary and save the variables
        tf.scalar_summary('loss_value', cost)
        tf.scalar_summary('accuracy', accuracy)
        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver(tf.all_variables())

        # initialize the tensorflow variables
        init = tf.initialize_all_variables()

        # load the train data object
        #train_data_obj = ImageData()

        # launch the graph
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                sess.run(init)

                # intializes weights and biases of the VGG 16 layers 
                if initialize_vgg_model:
                    logging.info(
                        "using the pretrained VGG model weights to intialize the layers")
                    initialize_conv_layers(
                        weights_file_path, conv_parameters=conv_parameters, sess=sess)
                    logging.info('weights and biases are intialized')

                # load pretrained model (different from the above initialization, it intializes the
                # model, using pretrained model on the new training data)
                if load_pretrained_model:
                    logging.info("loading pretrained model")
                    ckpt = tf.train.get_checkpoint_state('./checkpoints')
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        logging.info('model loaded')
                    else:
                        logging.info("no checkpoint found...")

                # write the tensorflow graph
                writer = tf.train.SummaryWriter('train_logs', sess.graph)
                tf.train.write_graph(
                    sess.graph_def, "./train_logs", 'train.pbtxt')
                logging.info("Session Initialized")
                
                # for epoch in range(n_epochs):
                #     # Keep training until reach max iterations
                #     for _ in xrange(train_data_obj.iter):
                #         try:
                #             # mnist.train.next_batch(batch_size)
                #             batch_x, batch_y = train_data_obj.next_batch()
                #             batch_y = np.array(batch_y, dtype=np.int32)
                #             # Fit training using batch data
                #             sess.run(optimizer, feed_dict={
                #                      x: batch_x, y: batch_y})

                #             # Calculate batch accuracy and loss
                #             acc = sess.run([accuracy, cost], feed_dict={
                #                            x: batch_x, y: batch_y})
                #             logging.info("Iter " + str(train_data_obj.batch_id) + ", Minibatch Loss= " + "{:.6f}".format(
                #                 acc[1]) + ", Training Accuracy= " + "{:.5f}".format(acc[0]))

                #         except Exception as e:
                #             logging.warn(" Training skipped at batch index %s" % (
                #                 train_data_obj.batch_id))
                #             print e

                #     # save the model after every epoch
                #     model_name = "model_%s_%s.ckpt" % (
                #         epoch, train_data_obj.batch_id)
                #     if not os.path.exists("checkpoints"):
                #         os.mkdir("checkpoints")

                #     checkpoint_path = saver.save(
                #         sess, os.path.join("checkpoints", model_name))
                #     logging.info("saving model %s" % checkpoint_path)

                #     # calculate the accuracy on the test data set
                #     test_accuracy = 0
                #     for tst_idx in xrange(num_test_batches):
                #         batch_x, batch_y = train_data_obj(tst_idx, mode="TEST")
                #         batch_y = np.array(batch_y, dtype=np.int32)
                #         # print "batch data generated"
                #         curr_accuracy = sess.run(accuracy, feed_dict={
                #                                  x: batch_x, y: batch_y})
                #         test_accuracy = np.add(test_accuracy, curr_accuracy)
                #         print "Test Accuracy for batch-%s: %s" % (tst_idx, curr_accuracy)

                #     logging.info("Overall test accuracy-%s" %
                #                  (np.divide(test_accuracy, num_test_batches)))

                
class ImageData(object):
    """load the pickled image data and generate batch data for training
    and testing the nodel
    """

    def __init__(self, mode='TRAIN'):
        logging.info('Loading {} data...'.format(mode))
        if mode == 'TRAIN':
            self.path = os.path.join(data_path, 'train')
            self.image_files = glob(os.path.join(self.path, '*', '*.jpg'))
        else:
            self.path = os.path.join(data_path, 'test')
            self.image_files = glob(os.path.join(self.path, '*.jpg'))
        random.shuffle(self.image_files)
        logging.info('Finished loading data..')
        self.size = len(self.image_files)
        self.batch_id = 0
        self.iter = int(self.size / batch_size)

    def next_batch(self):
        start = self.batch_id * batch_size
        end = (self.batch_id + 1) * batch_size
        self.batch_id += 1
        if self.batch_id >= self.iter:
            self.batch_id = 0
        X = []
        y = []
        for i in range(start, end):
            img = cv2.imread(self.image_files[i])
            img = cv2.resize(img, (img_height, img_width))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            X.append(img)
            label = int(self.image_files[i].split(os.path.sep)[-2][1])
            y.append(label)

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y


if __name__ == "__main__":
    data_path = "./"
    weights_file_path = "./vgg16_weights.npz"

    # cnn parameters
    batch_size = 16
    img_height = 224
    img_width = 224
    n_classes = 2 #[1=dog, 0=cat]
    learning_rate = 0.001
    n_img_channels = 3  # RGB image
    n_gap_channels = 1024
    n_epochs = 30
    display_step = 100
    num_test_batches = 0
    initialize_vgg_model = 1
    load_pretrained_model = 0
    VGG_MEAN = [103.939, 116.779, 123.68] #[B, G, R]


    # create the log file
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a stream handler and add that to the root logger
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # create a file handler
    logfilename = ''.join(str(datetime.now()).split('.')[0].split(':'))
    logging_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                '{}.log'.format(logfilename))
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(log_format)

    # add the file handler
    logger.addHandler(fh)
    # train the cnn model
    train()

