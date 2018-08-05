from __future__ import print_function
from __future__ import division

import numpy as np

import csv
import os
from six.moves import cPickle as pickle


def get_dataset_name(time_window, time_aggregation, forecast_window, forecast_aggregation, train_set_size,
                     valid_set_size):
    pickle_filename = 'dataset_'
    pickle_filename += str(time_window) + '_'
    pickle_filename += str(time_aggregation) + '_'
    pickle_filename += str(forecast_window) + '_'
    pickle_filename += str(forecast_aggregation) + '_'
    pickle_filename += str(train_set_size) + '_'
    pickle_filename += str(valid_set_size) + '.pickle'

    return pickle_filename


def parse_csv_file(filename, time_window, time_aggregation, forecast_window, forecast_aggregation):
    print('\tParsing', filename)

    timesteps = set()
    sections = set()
    data = []

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='\"')
        for row in reader:
            timesteps.add(int(row[0]))
            sections.add(int(row[1]))
            data.append(row[2:])

    data = np.asarray(data, dtype=np.float32)
    num_sections = max(sections) + 1
    num_timesteps = max(timesteps) + 1

    sequence = []

    for i in range(num_timesteps):
        stack = None
        for j in range(num_sections):
            stack = np.vstack([stack, data[i * num_sections + j]]) if stack is not None else data[i * num_sections + j]

        sequence.append(stack)

    d = []
    l = []

    max_timestep = num_timesteps - time_window * time_aggregation - forecast_window * forecast_aggregation + 1
    for i in range(0, max_timestep, time_aggregation):
        time_steps = []
        for j in range(time_window):
            initial_index = i + j * time_aggregation
            final_index = i + (j + 1) * time_aggregation
            time_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
        d.append(np.stack(time_steps, axis=1))
        forecast_steps = []
        for j in range(forecast_window):
            initial_index = i + time_window + j * forecast_aggregation
            final_index = i + time_window + (j + 1) * forecast_aggregation
            forecast_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
        l.append(np.stack(forecast_steps, axis=1))

    return d, l


def get_dataset(pickle_filename, args, parser):
    if os.path.exists(pickle_filename):
        print('Loading dataset from ' + pickle_filename + '...')

        with open(pickle_filename, 'rb') as f:
            save = pickle.load(f)
            valid_set = save['valid_set']
            valid_labels = save['valid_labels']
            valid_set2 = save['valid_set2']
            valid_labels2 = save['valid_labels2']
            test_set = save['test_set']
            test_labels = save['test_labels']
            mean = save['mean']
            stddev = save['stddev']
            f.close()

        train_set = np.load('train_set.npy')
        train_labels = np.load('train_labels.npy')
    else:
        if args.datasets is None or args.test_set is None:
            print('Dataset not found. You must give dataset and test set arguments from command line.')
            parser.print_help()
            exit()

        print('Generating training, validation and test sets...')

        dataset = []
        labels = []
        valid_set = []
        valid_labels = []
        test_set = []
        test_labels = []

        for dataset_file in args.datasets:
            ds, lb = parse_csv_file(dataset_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            dataset += ds
            labels += lb

        for valid_set_file in args.valid_set:
            ds, lb = parse_csv_file(valid_set_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            valid_set += ds
            valid_labels += lb

        for test_set_file in args.test_set:
            ds, lb = parse_csv_file(test_set_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            test_set += ds
            test_labels += lb

        del ds, lb

        permutation = np.random.permutation(len(dataset))
        dataset = np.asarray(dataset)[permutation]
        labels = np.asarray(labels)[permutation]
        permutation = np.random.permutation(len(valid_set) + len(test_set))
        valid_set = np.asarray(valid_set + test_set)[permutation]
        valid_labels = np.asarray(valid_labels + test_labels)[permutation]
        test_set = np.asarray(test_set)
        test_labels = np.asarray(test_labels)

        mean = np.mean(dataset, axis=(0, 1, 2))
        stddev = np.std(dataset, axis=(0, 1, 2))

        train_set = dataset[:args.train_set_size]
        train_labels = labels[:args.train_set_size]
        valid_set2 = valid_set[args.valid_set_size:2 * args.valid_set_size]
        valid_labels2 = valid_labels[args.valid_set_size:2 * args.valid_set_size]
        valid_set = valid_set[:args.valid_set_size]
        valid_labels = valid_labels[:args.valid_set_size]
        test_set = test_set[:args.valid_set_size]
        test_labels = test_labels[:args.valid_set_size]

        print('Saving dataset into ' + pickle_filename + '...')

        save = {
            'valid_set': valid_set,
            'valid_labels': valid_labels,
            'valid_set2': valid_set2,
            'valid_labels2': valid_labels2,
            'test_set': test_set,
            'test_labels': test_labels,
            'mean': mean,
            'stddev': stddev
        }

        f = open(pickle_filename, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        np.save('train_set.npy', train_set)
        np.save('train_labels.npy', train_labels)

    del save
    return (train_set, train_labels, valid_set, valid_labels, valid_set2, valid_labels2, test_set, test_labels, mean,
            stddev)


def MAE(labels, prediction):
    return np.mean(np.absolute(labels - prediction))


def MAPE(labels, prediction):
    return (np.mean(np.absolute(labels - prediction) / labels)) * 100


def MSE(labels, prediction):
    return np.mean(np.square(labels - prediction))


def RMSE(labels, prediction):
    return np.sqrt(MSE(labels, prediction))
