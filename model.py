from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim


def residual_block(feature_maps, x, previous_x, scope, is_training, reuse):
    a = slim.conv2d(x, feature_maps, [3, 3], [1, 1], activation_fn=None, scope=scope + '_1')
    a = slim.batch_norm(a, activation_fn=tf.nn.relu, updates_collections=None, is_training=is_training,
                        reuse=reuse, scope=scope + '_bn1')
    a = slim.conv2d(a, feature_maps, [3, 3], [1, 1], activation_fn=None, scope=scope + '_2')
    a = slim.batch_norm(a, activation_fn=None, updates_collections=None, is_training=is_training,
                        reuse=reuse, scope=scope + '_bn2')
    a += previous_x  # Residual connection
    return tf.nn.relu(a)


def model(x, mean, stddev, output_size, is_training=True, reuse=None):
    net = (x - mean) / stddev  # Normalization of input
    with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d], padding='SAME'):
            # Convolution layers
            previous_x = tf.pad(net, [[0, 0], [0, 0], [0, 0], [14, 15]])
            net = residual_block(32, net, previous_x, 'conv_1', is_training, reuse)
            net = residual_block(32, net, net, 'conv_2', is_training, reuse)
            net = residual_block(32, net, net, 'conv_3', is_training, reuse)

            previous_x = tf.pad(net, [[0, 0], [0, 0], [0, 0], [16, 16]])
            net = residual_block(64, net, previous_x, 'conv_4', is_training, reuse)
            net = residual_block(64, net, net, 'conv_5', is_training, reuse)
            net = residual_block(64, net, net, 'conv_6', is_training, reuse)

            previous_x = tf.pad(net, [[0, 0], [0, 0], [0, 0], [16, 16]])
            net = residual_block(96, net, previous_x, 'conv_7', is_training, reuse)
            net = residual_block(96, net, net, 'conv_8', is_training, reuse)
            net = residual_block(96, net, net, 'conv_9', is_training, reuse)

            net = slim.flatten(net)

        net = slim.fully_connected(net, 2048, scope='fc1')
        net = slim.dropout(net, keep_prob=0.6, is_training=is_training)
        net = slim.fully_connected(net, 1024, scope='fc2')
        out = slim.fully_connected(net, output_size, activation_fn=None, scope='fc_out')

    return out
