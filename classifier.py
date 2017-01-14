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
import logging
from datetime import datetime
import scipy.ndimage
import tensorflow as tf
import tensorflow.python.platform
from glob import glob
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner
import adamax

# cnn parameters
batch_size = 8
img_height = 150
img_width = 150
# n_classes = 2 #[1=dog, 0=cat]
n_img_channels = 3  # RGB image
n_epochs = 5
display_step = 100
# num_test_batches = 0
# data_path = "./"

# Global constants describing the data set.
weights_file_path = "./vgg16_weights.npz"
learning_rate = 0.0001
IMAGE_SIZE = 150
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_DEPTH = 3

NUM_CLASSES = 2 #[1=dog, 0=cat]
total_train_images = 25000
num_train_images = 20000
num_val_images = 5000

INITIALIZE_VGG_MODEL = 1  # to intialize vgg model with weights trained on imagenet
LOAD_PRETRAINED_MODEL = 0 # to initialize VGG model with  weights trained on dogs and cats images
VGG_MEAN = [103.939, 116.779, 123.68] #[B, G, R]

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './',
                           """Path to the data-set directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

tf.app.flags.DEFINE_string('train_dir', './train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', int(num_train_images/batch_size)*n_epochs,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



def conv_layer(input_fmaps, conv_parameters, kh, kw, n_out_fmaps, name, get_params, y_stride=1, x_stride=1 ):
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
    # init_range = math.sqrt(
    #     6.0 / (kh * kw * n_in_fmaps + kh * kw * n_out_fmaps))

    with tf.name_scope(name) as scope:
        # kernel_init_val = tf.random_uniform(
        #     [kh, kw, n_in_fmaps, n_out_fmaps], dtype=tf.float32, minval=-init_range, maxval=init_range)
        kernel = tf.get_variable("{}weights".format(scope), shape=[kh, kw, n_in_fmaps, n_out_fmaps],
         dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=False)        
        # kernel = tf.Variable(kernel_init_val, trainable=False, name='w')
        bias_init_val = tf.constant(0.0, shape=[n_out_fmaps], dtype=tf.float32)
        bias = tf.Variable(bias_init_val, trainable=False, name='b')
        out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_fmaps, kernel, strides=[
                         1, y_stride, x_stride, 1], padding='SAME'), bias))

        if get_params:
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



def fc_layer(input_fmaps, conv_parameters, n_out, name, get_params):
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
    # init_range = math.sqrt(6.0 / (n_in + n_out))

    with tf.name_scope(name) as scope:
        # kernel_init_val = tf.random_uniform(
        #     [n_in, n_out], dtype=tf.float32, minval=-init_range, maxval=init_range)
        # kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        kernel = tf.get_variable("{}weights".format(scope), shape=[n_in, n_out], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        bias = tf.Variable(bias_init_val, trainable=True, name='b')
        out = tf.add(tf.matmul(input_fmaps, kernel), bias)

        if get_params:
            conv_parameters +=[kernel, bias]

    return out


def conv_model(img, conv_parameters, n_classes, get_params=0):
    """Convolution neural net model (VGG model)
    Args:
    img: input image
    n_classes: number of output classes
    conv_parameters: list of all the convolution layers parameters (weights and biases)

    Returns:
    out: output of the final fully connected layer (apply softmax to it)
    """
    conv1_1 = conv_layer(img, conv_parameters, kh=3, kw=3,
                         n_out_fmaps=64, name='conv1_1', get_params=get_params)
    conv1_2 = conv_layer(conv1_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=64, name='conv1_2', get_params=get_params)
    pool1 = max_pool(conv1_2, name='pool1')

    conv2_1 = conv_layer(pool1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=128, name='conv2_1', get_params=get_params)
    conv2_2 = conv_layer(conv2_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=128, name='conv2_2', get_params=get_params)
    pool2 = max_pool(conv2_2, name='pool2')

    conv3_1 = conv_layer(pool2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_1', get_params=get_params)
    conv3_2 = conv_layer(conv3_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_2', get_params=get_params)
    conv3_3 = conv_layer(conv3_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=256, name='conv3_3', get_params=get_params)
    pool3 = max_pool(conv3_3, name='pool3')

    conv4_1 = conv_layer(pool3, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_1', get_params=get_params)
    conv4_2 = conv_layer(conv4_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_2', get_params=get_params)
    conv4_3 = conv_layer(conv4_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv4_3', get_params=get_params)
    pool4 = max_pool(conv4_3, name='pool4')

    conv5_1 = conv_layer(pool4, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_1', get_params=get_params)
    conv5_2 = conv_layer(conv5_1, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_2', get_params=get_params)
    conv5_3 = conv_layer(conv5_2, conv_parameters, kh=3,
                         kw=3, n_out_fmaps=512, name='conv5_3', get_params=get_params)
    pool5 = max_pool(conv5_3, name='pool5')
    
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    reshaped_layer = tf.reshape(pool5, [-1, flattened_shape], name="flat_pool")

    fc6 = fc_layer(reshaped_layer, conv_parameters, n_out=4096, name='fc6', get_params=get_params)
    with tf.name_scope(name='relu6') as scope:
        relu6 = tf.nn.relu(fc6)#, name='relu6')

    fc7 = fc_layer(relu6, conv_parameters, n_out=4096, name='fc7', get_params=get_params)
    with tf.name_scope(name='relu7') as scope:
        relu7 = tf.nn.relu(fc7)#, name='relu7')

    fc8 = fc_layer(relu7, conv_parameters, n_out=n_classes, name='fc8', get_params=get_params)

    return fc8


def loss_function(fc_last_layer, y):
    """loss function

    Args:
    fc_last_layer: the final layer of the convolution model
    y: ground truth labels

    Returns:
    cost: tensorflow loss variable

    """
    # generate one hot vectors for the ground truth labels
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, y])
    one_hot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)

    # apply softmax function on the and
    # calcualte loss using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        fc_last_layer, one_hot_labels)
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
    n_layers = 13*2 # (kernel and bias for each layer, except the last layer)

    if sess and conv_parameters:
        for i, k in enumerate(keys[:n_layers]):
            logging.info("{}, {}, {}".format(i, k, np.shape(weights[k])))
            sess.run(conv_parameters[i].assign(weights[k]))


def train():
    """ Intializes the convolution model with global average pooling and trains it on the
    input data

    """
    all_variables = ['conv1_1/weights:0', 'conv1_1/b:0', 'conv1_2/weights:0', 'conv1_2/b:0', 'conv2_1/weights:0', 
     'conv2_1/b:0', 'conv2_2/weights:0', 'conv2_2/b:0', 'conv3_1/weights:0', 'conv3_1/b:0', 'conv3_2/weights:0',
     'conv3_2/b:0', 'conv3_3/weights:0', 'conv3_3/b:0', 'conv4_1/weights:0', 'conv4_1/b:0', 'conv4_2/weights:0',
     'conv4_2/b:0', 'conv4_3/weights:0', 'conv4_3/b:0', 'conv5_1/weights:0', 'conv5_1/b:0', 'conv5_2/weights:0',
     'conv5_2/b:0', 'conv5_3/weights:0', 'conv5_3/b:0', 'fc6/weights:0', 'fc6/b:0', 'fc7/weights:0', 'fc7/b:0',
     'fc8/weights:0', 'fc8/b:0']    
    # generate the tensorflow computation graph
    with tf.Graph().as_default():
        # tf graph input
        # x = tf.placeholder(
        #     'float', [None, img_height, img_width, n_img_channels], name="InputData")
        # y = tf.placeholder(tf.int32, [None], name="Labels")

        # Get images and labels for the dataset
        x, y = inputs(data_dir=FLAGS.train_dir, batch_size=FLAGS.batch_size)
        val_x, val_y = inputs(data_dir=FLAGS.train_dir, batch_size=FLAGS.batch_size, TRAIN=0)

        # contruct CNN model
        conv_parameters = []  # list of cnn layers parameters
        cnn_out = conv_model(x, conv_parameters, n_classes=NUM_CLASSES, get_params=1)
        val_out = conv_model(x, [], n_classes=NUM_CLASSES, get_params=0)

        # define loss and optimizer
        cost = loss_function(cnn_out, y)
        val_cost = loss_function(val_out, val_y)
        optimizer = adamax.AdamaxOptimizer(
            learning_rate=learning_rate).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(
            # learning_rate=learning_rate).minimize(cost)

        # evaluate model
        correct_pred = tf.equal(tf.cast(tf.argmax(cnn_out, 1), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        val_correct_pred = tf.equal(tf.cast(tf.argmax(val_out, 1), tf.int32), val_y)
        val_accuracy = tf.reduce_mean(tf.cast(val_correct_pred, tf.float32))

        # create training summary and save the variables
        tf.scalar_summary('train_loss_value', cost)
        tf.scalar_summary('train_accuracy', accuracy)
        tf.scalar_summary('val_loss_value', val_cost)
        tf.scalar_summary('val_accuracy', val_accuracy)

        summary_op = tf.merge_all_summaries()
        file_name ="./checkpoints/model.ckpt-0.meta"
        if LOAD_PRETRAINED_MODEL:
            logging.info("loading pretrained model")
            ckpt = tf.train.get_checkpoint_state('./checkpoints')
            ckpt_path = ckpt.model_checkpoint_path
            print ckpt_path
            if ckpt and ckpt.model_checkpoint_path:
                reader = tf.train.NewCheckpointReader(ckpt_path)
                restore_dict = dict()
                for v in tf.trainable_variables():
                    tensor_name = v.name.split(':')[0]
                    if reader.has_tensor(tensor_name) and tensor_name not in ['fc8/weights', 'fc8/b']:
                        print('has tensor ', tensor_name)
                        restore_dict[tensor_name] = v

            saver = tf.train.Saver(restore_dict)
        else:
            # initialize the tensorflow variables
            saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()    
        # load the train data object
        #train_data_obj = ImageData()

        # launch the graph
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                sess.run(init)

                # intializes weights and biases of the VGG 16 layers 
                if INITIALIZE_VGG_MODEL:
                    logging.info(
                        "using the pretrained VGG model weights to intialize the layers")
                    initialize_conv_layers(
                        weights_file_path, conv_parameters=conv_parameters, sess=sess)
                    logging.info('weights and biases are intialized')

                if LOAD_PRETRAINED_MODEL:
                    saver.restore(sess, ckpt_path)                        

                # write the tensorflow graph
                summary_writer = tf.train.SummaryWriter('train_logs', sess.graph)
                tf.train.write_graph(
                    sess.graph_def, "./train_logs", 'train.pbtxt')
                logging.info("Session Initialized")

                # Start the queue runners.
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess=sess, coord=coord)
                try:
                    while not coord.should_stop():
                        for step in xrange(FLAGS.max_steps):
                          start_time = time.time()
                          sess.run(optimizer)
                          acc, loss_value = sess.run([accuracy, cost])
                          duration = time.time() - start_time

                          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                          logging.info("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
                              loss_value) + ", Training Accuracy= " + "{:.5f}".format(acc))


                          if (step+1) % 500 == 0:
                            # calculate validation accuracy
                            # Get images and labels for the dataset
                            for _ in xrange(int(num_val_images/batch_size)):
                                val_acc, val_loss_value = sess.run([val_accuracy, val_cost])
                                logging.info("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
                                    val_loss_value) + ", validation Accuracy= " + "{:.5f}".format(val_acc))                            
                            summary_str = sess.run(summary_op)
                            summary_writer.add_summary(summary_str, step)

                          # Save the model checkpoint periodically.
                          if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                            if not os.path.exists("checkpoints"):
                                os.mkdir("checkpoints")                    
                            checkpoint_path = os.path.join('checkpoints', 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)
                
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')

                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                   


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_contents, channels=3)
    # depth_major = tf.reshape(image, [IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH])
    # image = tf.transpose(depth_major, [1, 2, 0])
    # resized_image = tf.image.resize_image_with_crop_or_pad(image,
    #                                                        IMG_WIDTH, IMG_HEIGHT)
    resized_image = tf.image.resize_images(image,(256, 256))
    return resized_image, label


def read_labeled_image_list(data_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    filenames = glob(os.path.join(data_dir, "*.jpg"))
    # random.shuffle(filenames)
    labels = []
    for f in filenames:
        labels.append([1 if 'dog' in  f else 0][0])
        f = os.path.join(data_dir, f)
    return filenames, labels


def inputs(data_dir, batch_size, TRAIN=1):
    """
    Args:
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
  
    image_list, label_list = read_labeled_image_list(data_dir)
    if TRAIN:
        images = image_list[:num_train_images]
        labels = label_list[:num_train_images]
        # labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

        image, label = read_images_from_disk(input_queue)
        image = tf.cast(image, tf.float32)

        # Image processing for evaluation.
        # Randomly crop and flip the image horizontally.
        distorted_image = tf.random_crop(image, (IMG_HEIGHT, IMG_WIDTH, n_img_channels))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                         max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                       lower=0.2, upper=1.8)
        #Subtract off the mean and divide by the variance of the pixels.
        if INITIALIZE_VGG_MODEL:
            # Convert RGB to BGR
            float_image = tf.reshape(distorted_image, [img_width, img_height, n_img_channels], name='reshape_input_image')
            red, green, blue = tf.split(2, 3, float_image)
            assert red.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            assert green.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            assert blue.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            float_image = tf.concat(2, [
               blue - VGG_MEAN[0],
               green - VGG_MEAN[1],
               red - VGG_MEAN[2]])
        else:
            float_image = tf.image.per_image_standardization(distorted_image)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        X, y = tf.train.shuffle_batch([float_image, label], batch_size=batch_size, num_threads=16, capacity=capacity,
                                     min_after_dequeue=min_after_dequeue)
        y = tf.reshape(y, [FLAGS.batch_size], name='y')

    else:
        images = image_list[num_train_images:]
        labels = label_list[num_train_images:]
        # images = ops.convert_to_tensor(images, dtype=dtypes.string)
        # labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=False, capacity=1)

        image, label = read_images_from_disk(input_queue)
        image = tf.cast(image, tf.float32)        
        image = tf.image.resize_images(image,(IMG_HEIGHT, IMG_WIDTH))
        # float_image = tf.image.per_image_standardization(image)
        #Subtract off the mean and divide by the variance of the pixels.
        if INITIALIZE_VGG_MODEL:
            # Convert RGB to BGR
            float_image = tf.reshape(image, [img_width, img_height, n_img_channels], name='reshape_input_image')
            red, green, blue = tf.split(2, 3, float_image)
            assert red.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            assert green.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            assert blue.get_shape().as_list() == [IMG_HEIGHT, IMG_WIDTH, 1]
            float_image = tf.concat(2, [
               blue - VGG_MEAN[0],
               green - VGG_MEAN[1],
               red - VGG_MEAN[2]])
        else:
            float_image = tf.image.per_image_standardization(image)
        
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size

        X,y = tf.train.batch([float_image, label], batch_size=batch_size, num_threads=16)
        y = tf.reshape(y, [FLAGS.batch_size], name='y_val')        


    return X, y


if __name__ == "__main__":

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



