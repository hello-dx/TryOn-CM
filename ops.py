import numpy as np
import tensorflow as tf
from PIL import Image


def batch_norm(x, epsilon=1e-5, momentum = 0.9, name=None, is_training=True):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, scope='bn', is_training=is_training)

def conv2d(input_, output_dim,  k=5, d=2, stddev=0.02, name="conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d, d, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def deconv2d(input_, output_shape, k=5, d=2, stddev=0.02, name="deconv2d", bias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d, d, 1])
        if bias:
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)
        return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def load_image(img_path, img_size=256):
    img = np.array(Image.open(img_path).resize([img_size, img_size]))
    img = img / 127.5 - 1.
    return img.astype(np.float32)

