from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

import argparse
import os
import time

import utils
from model import model

parser = argparse.ArgumentParser(description='Trains a convolutional network for traffic prediction.')
files_group = parser.add_argument_group('Data files')
files_group.add_argument('-d', '--datasets', type=str, help='list of files to make training and validation sets',
                         nargs='+', metavar=('FILE 1', 'FILE 2'))
files_group.add_argument('-v', '--valid_set', type=str, help='list of validation set files', nargs='+',
                         metavar=('FILE 1', 'FILE 2'))
files_group.add_argument('-t', '--test_set', type=str, help='file of the test set data', nargs='+',
                         metavar=('FILE 1', 'FILE 2'))
prediction_group = parser.add_argument_group('Prediction parameters')
prediction_group.add_argument('-tw', '--time_window', default=12, type=int, help='time window used to predict')
prediction_group.add_argument('-ta', '--time_aggregation', default=1, type=int, help='steps aggregated for net input')
prediction_group.add_argument('-fw', '--forecast_window', default=1, type=int, help='time window to be predicted')
prediction_group.add_argument('-fa', '--forecast_aggregation', default=1, type=int, help='steps aggregated in forecast')
training_group = parser.add_argument_group('Training parameters')
training_group.add_argument('-ts', '--train_set_size', default=70000, type=int, help='training set size')
training_group.add_argument('-vs', '--valid_set_size', default=30000, type=int, help='validation set size')
training_group.add_argument('-vp', '--valid_partitions', default=100, type=int, help='validation set partitions number')
training_group.add_argument('-tp', '--test_partitions', default=100, type=int, help='test set partitions number')
training_group.add_argument('-b', '--batch_size', default=70, type=int, help='batch size for SGD')
training_group.add_argument('-l', '--learning_rate', default=1e-4, type=float, help='learning rate for SGD')
training_group.add_argument('-dr', '--decay_rate', default=0.1, type=float, help='learning rate decay rate')
training_group.add_argument('-ds', '--decay-steps', default=1000, type=int, help='learning rate decay steps')
training_group.add_argument('-c', '--gradient_clip', default=40.0, type=float, help='clip at this max norm of gradient')
training_group.add_argument('-m', '--max_steps', default=10000, type=int, help='max number of iterations for training')
training_group.add_argument('-s', '--save', action='store_true', help='save the model every epoch')
training_group.add_argument('-ens', '--ensemble', default=1, type=int, help='Number of the model in the ensemble')
args = parser.parse_args()

pickle_filename = utils.get_dataset_name(args.time_window, args.time_aggregation, args.forecast_window,
                                         args.forecast_aggregation, args.train_set_size, args.valid_set_size)

dataset = utils.get_dataset(pickle_filename, args, parser)
train_set = dataset[0]
train_labels = dataset[1]
valid_set = dataset[2]
valid_labels = dataset[3]
valid_set2 = dataset[4]
valid_labels2 = dataset[5]
test_set = dataset[6]
test_labels = dataset[7]
mean = dataset[8]
stddev = dataset[9]
del dataset

print('Training set', train_set.shape, train_labels.shape)
print('Validation set', valid_set.shape, valid_labels.shape)
print('Test set', valid_set2.shape, valid_labels2.shape)

print('Building model...')

graph = tf.Graph()
with graph.as_default():
    # Input data
    tf_train_set = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, train_set.shape[1], args.time_window, 3))
    tf_train_labels_q = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 1))
    tf_valid_set = tf.placeholder(dtype=tf.float32, shape=(None, train_set.shape[1], args.time_window, 3))
    tf_valid_labels_q = tf.placeholder(dtype=tf.float32, shape=(None, 1, args.forecast_window, 1))
    tf_test_set = tf.placeholder(dtype=tf.float32, shape=(None, train_set.shape[1], args.time_window, 3))
    tf_test_labels_q = tf.placeholder(dtype=tf.float32, shape=(None, 1, args.forecast_window, 1))
    tf_mean = tf.constant(mean)
    tf_stddev = tf.constant(stddev)

    # Training computation
    output = model(tf_train_set, tf_mean, tf_stddev, args.forecast_window)
    s = tf_train_labels_q.get_shape().as_list()
    desired_output = tf_train_labels_q
    loss = tf.nn.l2_loss(output - desired_output)

    # Optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_steps, args.decay_rate,
                                               staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, global_norm = tf.clip_by_global_norm(grads, args.gradient_clip)  # Gradient clipping
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Predictions for the validation and test sets
    q_tr_error = tf.reduce_mean(tf.abs(output - desired_output))
    valid = model(tf_valid_set, tf_mean, tf_stddev, args.forecast_window, is_training=False, reuse=True)
    q_valid_error = tf.reduce_mean(tf.abs(valid - tf_valid_labels_q))
    test = model(tf_test_set, tf_mean, tf_stddev, args.forecast_window, is_training=False, reuse=True)
    q_test_error = tf.reduce_mean(tf.abs(test - tf_test_labels_q))

    # Saver object to save and restore TensorFlow variables
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    if os.path.exists('model.ckpt'):
        print('Restoring model from checkpoint...')
        saver.restore(session, 'model.ckpt')
        args.max_steps = -1
    else:
        tf.initialize_all_variables().run()

    print('Initialized')

    v = train_labels.shape[1] // 2  # We predict a traffic variable for the road section in the middle

    t1 = time.time()
    for step in range(args.max_steps + 1):
        offset = (step * args.batch_size) % (train_labels.shape[0])
        batch_data = train_set[offset:offset + args.batch_size]
        batch_labels_q = train_labels[offset:offset + args.batch_size, v, :, 0]  # 0: Flow, 1: Occupancy, 2: Speed
        feed_dict = {tf_train_set: batch_data,
                     tf_train_labels_q: batch_labels_q}
        _, l2, ooo, lr = session.run(
            [optimizer, loss, output, learning_rate], feed_dict=feed_dict)
        q_e = utils.MAE(train_labels[offset:offset + args.batch_size, v, :, 0], ooo)
        t = (time.time() - t1) * 1000
        t1 = time.time()
        if step % (train_set.shape[0] / (args.batch_size * 10)) == 0:  # Give feedback 10 times in an epoch
            print('Step %d (epoch %.2f), %.1f ms' % (step, (step * args.batch_size / train_set.shape[0]), t))
            print('Minibatch loss: %f, learning rate: %f' % (l2, lr))
            print('Minibatch errors:')
            print('\tMAE: %.1f veh/h' % q_e)
        if step % (train_set.shape[0] / args.batch_size) == 0:  # Evaluate validation set every epoch
            q_e = 0.
            q_e_p = 0.
            q_e_m = 0.
            q_e_r = 0.

            v_p = []
            v_l = []
            for k in range(args.valid_partitions):
                v_batch_size = valid_set.shape[0] // args.valid_partitions
                feed_dict = {tf_valid_set: valid_set[k * v_batch_size:(k + 1) * v_batch_size],
                             tf_valid_labels_q: valid_labels[k * v_batch_size:(k + 1) * v_batch_size, v:v + 1, :, 0:1]}
                v_ooo = session.run(valid, feed_dict=feed_dict)
                v_p.append(v_ooo)
                v_l.append(valid_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0])
                v_q_e = utils.MAE(valid_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], v_ooo)
                q_e_p += utils.MAPE(valid_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], v_ooo)
                q_e_m += utils.MSE(valid_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], v_ooo)
                q_e += v_q_e

            q_e /= args.valid_partitions
            q_e_p /= args.valid_partitions
            q_e_m /= args.valid_partitions
            q_e_r = np.sqrt(q_e_m)

            print('=' * 80)
            print('Validation errors:')
            print('\tMAE: %f veh/h, (%.2f%%)' % (q_e, q_e_p))
            print('\tMSE: %f' % q_e_m)
            print('\tRMSE: %f' % q_e_r)
            print('=' * 80)

            # Save model into model.ckpt
            print('Saving model...')
            saver.save(session, 'model_' + str(args.ensemble))

            # Shuffle train set every epoch
            permutation = np.random.permutation(train_set.shape[0])
            train_set = train_set[permutation]
            train_labels = train_labels[permutation]

    q_e = 0.
    t_e_p = 0.
    t_e_m = 0.
    t_e_r = 0.
    te = None
    first_time = True

    v_p = []
    v_l = []
    for k in range(args.test_partitions):
        v_batch_size = valid_set2.shape[0] // args.test_partitions
        feed_dict = {tf_test_set: valid_set2[k * v_batch_size:(k + 1) * v_batch_size],
                     tf_test_labels_q: valid_labels2[k * v_batch_size:(k + 1) * v_batch_size, v:v + 1, :, 0:1]}
        t, v_q_e = session.run([test, q_test_error], feed_dict=feed_dict)
        v_p.append(t)
        v_l.append(valid_labels2[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0])
        t_e = utils.MAE(valid_labels2[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], t)
        t_e_p += utils.MAPE(valid_labels2[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], t)
        t_e_m += utils.MSE(valid_labels2[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], t)
        if first_time:
            te = t
            first_time = False
        else:
            te = np.vstack([te, t])
        q_e += t_e

    q_e /= args.test_partitions
    t_e_p /= args.test_partitions
    t_e_m /= args.test_partitions
    t_e_r = np.sqrt(t_e_m)

    print('\n\nTest errors:')
    print('\tMAE: %f veh/h (%.2f%%)' % (q_e, t_e_p))
    print('\tMSE: %f' % t_e_m)
    print('\tRMSE: %f' % t_e_r)

    q_e = 0.
    t_e_p = 0.
    te = None
    first_time = True

    for k in range(args.test_partitions):
        v_batch_size = test_set.shape[0] // args.test_partitions
        feed_dict = {tf_test_set: test_set[k * v_batch_size:(k + 1) * v_batch_size],
                     tf_test_labels_q: test_labels[k * v_batch_size:(k + 1) * v_batch_size, v:v + 1, :, 0:1]}
        t, v_q_e = session.run([test, q_test_error], feed_dict=feed_dict)
        t_e = utils.MAE(test_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], t)
        t_e_p += utils.MAPE(test_labels[k * v_batch_size:(k + 1) * v_batch_size, v, :, 0], t)
        if first_time:
            te = t
            first_time = False
        else:
            te = np.vstack([te, t])
        q_e += t_e

    q_e /= args.test_partitions
    t_e_p /= args.test_partitions

    np.save('test_prediction_' + str(args.ensemble) + '.npy', te)
    np.save('test_labels_' + str(args.ensemble) + '.npy', test_labels[:, v, :, 0])
