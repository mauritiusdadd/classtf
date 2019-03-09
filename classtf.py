#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify-tf is Random Forest Classifier based on tensorflow

Copyright 2018-2019 Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is released under MIT license

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

References:
- https://stackoverflow.com/questions/4854244
- https://www.listendata.com/2017/05/feature-selection-boruta-package.html
- http://blog.gradientmetrics.com/2017/10/30/boruta/
- https://www.jstatsoft.org/article/view/v036i11/v36i11.pdf
"""

import os
import sys
import time
import errno
import logging
import datetime
import argparse
import tempfile
import textwrap
import configparser
import numpy as np
import astropy as ap
from astropy.table import Table, Column

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from tensorflow.python.platform import tf_logging


VERSION_MAJ = 0
VERSION_MIN = 4
VERSION_REV = 2


# Ignore all GPUs, tf random forest does not benefit from it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CHECKPOINT_NAME = "checkpoint"
CHECKPOINT_INTERVAL = 180

MSG_ERR_FILE_NOT_FOUND = "Error: '{}' is not a valid file od does not exist"
MSG_ERR_FIMPO = "Error: feature importance can be determined only if "\
                "both train and a test datasets are specified using the "\
                "arguments --train and --test!"
MSG_ERR_DIR_NOT_FOUND = "Error: The directory '{}' does not exist"
MSG_ERR_DIR_PERM = "Error: you do not have permission to access {}"
MSG_ERR_TARGETCOL = "Error: target column '{}' does not exist"
MSG_WRN_EXCLUDE = "Warning: column '{}' does not exist"
MSG_INFO_MODELDIR = "Info: model directory is '{}'"

LOG_MODEL_INFO = """
Model configuration:
 - tree:    {0.num_trees: 6d}
 - depth:   {0.max_nodes: 6d}
 - classes: {0.num_classes: 6d}
 - features:{0.num_features: 6d}
"""

INFO_TRAIN_PROG = "Training {:s} tresh: {:.3f} acc.: {:.4f}"
INFO_TEST_PROG = "Testing  {:s}"


def timestr2sec(time_string):
    """
    Convert a time string to a time interval

    Parameters
    ----------
    time_string : str
        The string representing the time interval. Allowed units
        are y (years), d (days), h (hours), m (minutes), s (seconds)

    Returns
    -------
    delta_t : float
        The time interval in seconds


    Examples
    --------
    >>> timestr2sec("1d 2h 4.25s")
    93604.25

    >>> timestr2sec(None)
    0.0
    """

    if not time_string:
        return -1

    data = time_string.replace(' ', '')
    delta_t = 0

    try:
        years, data = data.split('y')
        delta_t += float(years)*31536000
    except ValueError:
        pass

    try:
        days, data = data.split('d')
        delta_t += float(days)*86400
    except ValueError:
        pass

    try:
        hours, data = data.split('h')
        delta_t += float(hours)*3600
    except ValueError:
        pass

    try:
        minustes, data = data.split('m')
        delta_t += float(minustes)*60
    except ValueError:
        pass

    try:
        seconds, data = data.split('s')
        delta_t += float(seconds)
    except ValueError:
        pass

    return delta_t


def pbar(val, maxval, empty='-', full='#', size=50):
    """
    return a string that represents a nice progress bar

    Parameters
    ----------
    val : float
        The fill value of the progress bar

    maxval : float
        The value at which the progress bar is full

    empty : str
        The character used to draw the empty part of the bar

    full : str
        The character used to draw the full part of the bar

    size : integer
        The lenght of the bar expressed in number of characters

    Returns
    -------
    br : str
        The string containing the progress bar

    Examples
    --------
    >>> a = pbar(35.5, 100, size=5)
    >>> print(a)
     35.50% [##----]
    """
    br = "{{1: 6.2f}}% [{{0:{}<{}s}}]".format(empty, size)
    br = br.format(full*int(size*val/maxval), val*100/maxval)
    return br


def _check_file(filename):
    """
    Check if a file exists

    Parameters
    ----------
    filename : str
        the name of the file to be checked

    Returns
    -------
        out : bool
            True if the filename is a file
            False if file bool(filename) is False
            exit the program otherewise
    """
    if filename:
        if not os.path.isfile(filename):
            print(MSG_ERR_FILE_NOT_FOUND.format(filename))
            sys.exit(1)
        return True
    return False


def readinput(filename, target_col=None, exclude_features=None,
              make_shadows=False):
    """
    Read a dataset and return an input and a target features subset

    Parameters
    ----------
        filename  : str
            The file path of the dataset to be read

        target_col : str or integer, optional (default None)
            The column name or index to be used as a target subset

        exclude_features : list of str, optional (default None)
            A list containing the name of the features that must be ignored

        make_shadows : bool, optional (default False)
            Generate a shawow copy for each feature that is not in
            the excluded list. The name of the shadow feature is the same
            of the original feature with the prefix '__shadow_'

    Returns
    -------
        x_data : ndarray
            The input subset

        y_data : ndarray
            The target subset
    """
    t = Table.read(filename)

    if exclude_features:
        try:
            t.remove_columns(exclude_features)
        except KeyError:
            for cname in exclude_features:
                try:
                    t.remove_column(cname)
                except KeyError:
                    print(MSG_WRN_EXCLUDE.format(cname))

    if target_col:
        try:
            colname = t.colnames[target_col]
        except TypeError:
            colname = target_col

        try:
            y_col = t.columns.pop(colname)
        except KeyError:
            print(MSG_ERR_TARGETCOL.format(colname))
            sys.exit(1)
        else:
            y_data = y_col.data.copy()
            y_names = colname
    else:
        y_data = None
        y_names = None

    if make_shadows:
        ncols = len(t.colnames)
        for i, col_name in enumerate(t.colnames[:]):
            #
            # NOTE: making a random shuffle on the column itself will
            #       most probably corrupt the table. We are forced thus
            #       to create a new column with already shuffled data.
            #
            #       well don't aske me why, this is probably a bug in
            #       astropy of numpy, but it seems the only way to shuffle
            #       column data is to use the fuction np.random.permutation
            #
            #       see: https://github.com/astropy/astropy/issues/6318
            shadow_data = np.random.permutation(t[col_name].data)
            shadow_name = '__shadow_' + col_name
            t.add_column(Column(shadow_data, shadow_name, 'float64'))
            print("Building shadows {}\r".format(pbar(i, ncols)), end='\r')
        print("Building shadows {}\n".format(pbar(1, 1)), end='\r')
        print("Built {} shadows for {} features\n\n".format(i, ncols))

    n = t.as_array()
    x_data = np.empty((len(t), len(t.colnames)), dtype=np.float64)
    for n, col in enumerate(t.colnames):
        x_data[:, n] = t[col].astype(np.float64)

    # x_data = n.view(np.float64).reshape(n.shape + (-1,))
    x_names = t.colnames.copy()

    return x_data, y_data, x_names, y_names


def score(cm):
    """
    Compute the precision, the recall and the f1-score
    for the given confusion matrix

    Parameters
    ----------
        cm : (n, n) ndarray
            The input confusion matrix

    Returns
    -------
        score : (n 3) ndarray
            The array contains the tripletes [purity, recall, f1-score]
            for each class of objects represented in the confusion matrix

    Examples
    --------
    >>> print(cm)
    [[31633.,     0.,     0.]
     [ 3494.,  8902.,     0.]
     [    0.,   907.,  5064.]]

    >>> score(cm)
    [[0, 0.9005323540296638, 1.0, 0.9476632714200119],
     [1, 0.9075338974411254, 0.7181348822200709, 0.8018013960819635],
     [2, 1.0, 0.8480991458717133, 0.9178069777979158]]
    """
    scores = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = 0
        fn = 0
        for j in range(len(cm)):
            if j != i:
                fp += cm[j][i]
                fn += cm[i][j]
        purity = tp / (tp + fp)
        compl = tp / (tp + fn)
        f1 = 2 * purity*compl / (purity+compl)
        scores.append([i, purity, compl, f1])
    return scores


class RFModel():
    """
    Random Forset Classifier

    Attributes
    ----------
    num_trees : integer
        Number of trees of the model

    max_nodes : integer
        Max number of nodes

    see tf.contrib.tensor_forest.tensor_forest.ForestHParams
    """

    def __init__(self, num_trees=100, max_nodes=100000, tf_session=None,
                 loss_treshold=10, batch_size=1024, train_timeout=-1):
        """
        Initialize RFModel objects

        Parameters
        ----------
        num_trees : integer, optional (default 100)
            Number of trees to generate

        max_nodes : integer, optional (default 1000)
            Maximum number of nodes

        tf_session : tensorflow.Session, optional (default None)
            Tensorflow Session. In omitted a new session is created

        loss_treshold : float, optional (default 10)
            when the loss change bewteen two training cycles drops
            below this value then the trainng process is stopped

        batch_size : integers, optional (default 1024)
            Number of objects to digest per cycle

        train_timeout : integer, optional (default -1)
            Number of seconds after which the training proces
            shoudl be stopped
        """

        self.num_trees = num_trees
        self.max_nodes = max_nodes

        self.num_classes = None
        self.num_features = None

        self.loss_treshold = loss_treshold
        self.batch_size = int(batch_size)
        self.train_timeout = train_timeout

        self.model_info = None

        if tf_session is None:
            self.sess = tf.Session()

        else:
            self.sess = tf_session

    def buildmodel(self, num_classes, num_features):
        """
        Build the Random Forest Model

        Parameters
        ----------
        num_classes : integer
            number of target classes

        num_features : integer
            number of input features
        """
        print("Building RF model...")

        self.num_classes = num_classes
        self.num_features = num_features

        # Input data
        X = tf.placeholder(tf.float32, shape=[None, num_features])

        # For random forest, labels must be integers (the class id)
        Y = tf.placeholder(tf.int32, shape=[None])

        # Random Forest Parameters
        hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                              num_features=num_features,
                                              num_trees=self.num_trees,
                                              max_nodes=self.max_nodes,
                                              split_after_samples=200).fill()

        # Build the Random Forest
        forest_graph = tensor_forest.RandomForestGraphs(hparams)

        # Get training graph and loss
        train_op = forest_graph.training_graph(X, Y)
        loss_op = forest_graph.training_loss(X, Y)

        # Measure the accuracy
        infer_op, _, _ = forest_graph.inference_graph(X)
        correct_prediction = tf.equal(tf.argmax(infer_op, 1),
                                      tf.cast(Y, tf.int64))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Save parameters for later use
        self.model_info = {
            'X': X,
            'Y': Y,
            'OPS': {
                'infer': infer_op,
                'train': train_op,
                'loss': loss_op,
                'accuracy': accuracy_op
            }
        }

        # Initialize the variables and forest resources
        init_vars = tf.group(
            tf.global_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()))

        # Run the initializer
        self.sess.run(init_vars)

    def train(self, x_data, y_data, model_dir='.'):
        """
        Train the classifier model

        Parameters
        ----------
        x_data : array like
            An array containing the input features

        y_data : array like
            An array containing the terget classes

        model_dir : str, optional (default cwd)
            directory used for saving the model
        """

        model_file = os.path.join(model_dir, CHECKPOINT_NAME)

        # Compute the number of input features and target classes
        num_features = x_data.shape[-1]
        num_classes = len(set(y_data))

        config = configparser.ConfigParser()
        config['MODEL'] = {
            'features': str(num_features),
            'classes': str(num_classes),
            'depth': str(self.max_nodes),
            'num_trees': str(self.num_trees),
            'batch_size': str(self.batch_size),
            'loss_treshold': str(self.loss_treshold),
            'train_timeout': str(self.train_timeout)
        }
        with open(os.path.join(model_dir, "model.params"), "w") as outf:
            config.write(outf)
        del config

        # Build the model
        self.buildmodel(num_classes, num_features)
        print("Training...", end='\r')

        # Create the saver interface used save the trained model
        saver = tf.train.Saver(save_relative_paths=True,
                               filename='checkpont-state',
                               max_to_keep=10)

        # Load placeholders
        X = self.model_info['X']
        Y = self.model_info['Y']

        # Load operations
        train_op = self.model_info['OPS']['train']
        loss_op = self.model_info['OPS']['loss']
        accuracy_op = self.model_info['OPS']['accuracy']

        # Validation dataset
        v_size = self.batch_size
        b_valid_x = x_data[:self.batch_size]
        b_valid_y = y_data[:self.batch_size]

        # Training loop
        dlen = len(x_data)
        b_start = v_size
        b_end = b_start + self.batch_size
        i = 0
        batch = {}

        # train the model until the loss treshold is reached or the
        # timeout is expired
        t_start = time.time()
        timed_out = False
        old_loss = 9999
        delta_loss = self.loss_treshold + 9999
        while (delta_loss > self.loss_treshold) and not timed_out:
            if b_start < dlen:
                # While we have unused input data get a new batch...
                batch_x = x_data[b_start:b_end]
                batch_y = y_data[b_start:b_end]
            else:
                # If all data have been alreay used, then create a random
                # subset from the whole training dataset (except the
                # validation subset
                indeces = np.random.rand(self.batch_size)*(dlen - v_size)
                indeces = list(np.floor(indeces + v_size))
                batch_x = x_data.take(indeces, axis=0)
                batch_y = y_data.take(indeces, axis=0)

            b_start += self.batch_size
            b_end = b_start+self.batch_size

            # ...and feed it to the model
            batch[X] = batch_x
            batch[Y] = batch_y

            _, j = self.sess.run([train_op, loss_op], feed_dict=batch)

            if i == 0:
                old_loss = j
            else:
                delta_loss = 100*(j - old_loss)/old_loss
                old_loss = j

            if i % 10 == 0:
                # Every ten cycles compute the accuracy using the validation
                # dataset
                feed_dict = {X: b_valid_x, Y: b_valid_y}
                acc = self.sess.run(accuracy_op, feed_dict=feed_dict)

                # computing the normalized training progress
                progress = 1 / (np.log(delta_loss/self.loss_treshold) + 1)

                # computing the elapsed time
                delta_t = time.time() - t_start
                if self.train_timeout > 0:
                    if delta_t > self.train_timeout:
                        timed_out = True
                    else:
                        progress = max(delta_t/self.train_timeout, progress)

                # then update the progress bar
                my_bar = pbar(progress, 1)
                msg_str = INFO_TRAIN_PROG.format(my_bar, delta_loss, 100*acc)
                print(msg_str, end='\r')

                if time.time() - t_start >= CHECKPOINT_INTERVAL:
                    # and every now and then save current model state
                    saver.save(self.sess, model_file, global_step=b_start)
                    t_start = time.time()

            i += 1

        # Save the fully trained model
        saver.save(self.sess, model_file, global_step=b_start)
        print(INFO_TRAIN_PROG.format(pbar(1, 1), j, acc))

    def run(self, x_data, model_dir=None):
        """
        Run the trained model

        Parameters
        ----------
        x_data : array like
            An array containing the input features

        model_dir : str, optional (default cwd)
            directory containing the trained model

        Returns
        -------
        pred : array of int32
            the predicted classes
        """

        # If there is no model then we should load it from a saved state
        if self.model_info is None:
            # Load model parameters
            config = configparser.ConfigParser()
            config.read(os.path.join(model_dir, "model.params"))
            num_features = int(config['MODEL']['features'])
            num_classes = int(config['MODEL']['classes'])
            self.max_nodes = int(config['MODEL']['depth'])
            self.num_trees = int(config['MODEL']['num_trees'])
            self.batch_size = int(config['MODEL']['batch_size'])
            self.loss_treshold = float(config['MODEL']['loss_treshold'])
            self.train_timeout = float(config['MODEL']['train_timeout'])
            del config

            # Build the model
            self.buildmodel(num_classes, num_features)

            # Load the trained model
            print("Loading saved state...")
            checkpoint_file = tf.train.latest_checkpoint(
                model_dir,
                CHECKPOINT_NAME)
            saver = tf.train.import_meta_graph(
                "{}.meta".format(checkpoint_file),
                clear_devices=True)
            saver.restore(self.sess, checkpoint_file)

        # Load placeholders
        X = self.model_info['X']

        # Load operations
        infer_op = self.model_info['OPS']['infer']

        # Testing loop
        i = 0
        pred = []

        total = len(x_data)
        b_start = 0
        b_end = self.batch_size
        batch = {}

        while b_start < total:
            # While we have unused input data get a new batch...
            batch[X] = x_data[b_start:b_end]

            b_start += self.batch_size
            b_end = b_start+self.batch_size

            # ...and feed it to the model
            res = self.sess.run(infer_op, feed_dict=batch)

            # probabilities += list(res)
            pred += list(np.argmax(res, axis=1))

            if i % 10 == 0:
                # Every ten cycle update the progress bar
                print(INFO_TEST_PROG.format(pbar(b_start, total)), end='\r')
            i += 1

        print(INFO_TEST_PROG.format(pbar(1, 1)))
        return np.array(pred, dtype='int32')

    def test(self, x_data, y_data, model_dir=None):
        """
        Test the trained model

        Parameters
        ----------
        x_data : array like
            An array containing the input features

        y_data : array like
            An array containing the target classes

        model_dir : str, optional (default cwd)
            directory containing the trained model

        Returns
        -------
        pred : array of int32
            the predicted classes

        cm : array like
            the confusion matrix
        """
        y_data = np.array(y_data, dtype='int32')
        pred = self.run(x_data, model_dir)
        # Compute the confusion matrix
        cm = self.sess.run(tf.confusion_matrix(pred, y_data))
        return pred, cm


def _main():
    # commandline argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Random Forest Classifier based on Tensorflow (TM)',
        epilog=textwrap.dedent('''\
            Notes:
              You *MUST* specify at least one of the -r, -t or -x
              options

            Examples:

              - Simple training and testing:
                %(prog)s -t cat1.fits -x cat2.fitsi -c "Class"

              - Train the classifier and save the trained model:
                %(prog)s --train traincat.fits --model-dir ./mymodel/

              - Load saved model and run it on a catalogl:
                %(prog)s --model-dir ./mymodel --run mycat.fits

              - Train the classifier with a timeout of 1 day and 6 hours
                %(prog)s --train traincat.votable --train_timeout 1d3h
                %(prog)s --train traincat.votable --train_timeout 1.25d
                %(prog)s --train traincat.votable --train_timeout 30h

              - Train the classifier with a loss treshold of 0.001
                %(prog)s --train train.csv --loss-treshold 0.001

            For more info import the classtf module in python and run
            >>> help(classtf)


            '''))

    parser.add_argument('-r', '--run', type=str, action='store',
                        metavar='FILE', dest='run_filename',
                        help="If specified, the classifier is run using\
                              the dataset %(metavar)s as input")

    parser.add_argument('-t', '--train', type=str, action='store',
                        metavar='FILE', dest='train_filename',
                        help="If specified, the classifier is trained using\
                              the dataset %(metavar)s")

    parser.add_argument('-x', '--test', type=str, action='store',
                        metavar='FILE', dest='test_filename',
                        help="If specified, the classifier is tested using\
                              the dataset %(metavar)s")

    parser.add_argument('-f', '--feature-importance', action='store_true',
                        dest='do_feature_importance',
                        help="If specified, the importance of each feature\
                              is computed. Can be used only if both --train\
                              and --test options are specified")

    parser.add_argument('--loss-treshold', type=float,  action='store',
                        metavar='VALUE', dest='loss_treshold', default=0.001,
                        help="If specified, the training will stop when the\
                              loss changes between two cycles becomes smaller\
                              than %(metavar)s (or when the training timeout\
                              expires). This option has effect only when the\
                              option --train is specified and is ignored\
                              otherwise. If not specified, the default value\
                              of 0.001 is used.")

    parser.add_argument('--train-timeout', type=str, action='store',
                        metavar='TIME_INTERVAL', dest="train_timeout",
                        help="If specified set the maximum execution time\
                              for the training process. %(metavar)s must be\
                              a string representing a time interval. Allowed\
                              units are y (years), d (days), h (hours),\
                              m (minutes), s (seconds) [i.e. 1y2d13h20m13.3s\
                              If not specified, no timeout is applied")

    parser.add_argument('-c', '--target', type=str, action='store',
                        metavar='TARGET_FEATURE_ID', dest='class_target',
                        default=-1,
                        help="Set the name or column index of feature used as\
                              target class during training and testing. If\
                              not specified, the last column in the dataset\
                              is used as default")

    parser.add_argument('--ignore-features', type=str, action='store',
                        metavar='', dest='skip_list', nargs='+',
                        help="List of features that should be ignored")

    parser.add_argument('-d', '--model-dir', type=str, action='store',
                        metavar='MODEL_DIR', dest='model_dir',
                        help="If specified, the trained model is saved or\
                              restored from %(metavar)s")

    parser.add_argument('-n', '--trees', type=int, action='store',
                        metavar='NUM_OF_TREES', dest='num_trees',
                        default=1000,
                        help="If specified, set the number of generated trees\
                              to %(metavar)s. Otherwisee fallback to the\
                              default value of 1000 trees")

    parser.add_argument('-b', '--batch-size', type=int, action='store',
                        metavar='BATCH_SIZE', dest='batch_size',
                        default=4096,
                        help="If specified, set  the size of the batch to\
                              to %(metavar)s, which is the number of object\
                              used at once during a training/test/run cycle.\
                              The default value is 4096")

    parser.add_argument('--depth', type=int, action='store',
                        metavar='NUM_NODES', dest='max_nodes',
                        default=10000,
                        help="If specified, set  the maximum number of nodes\
                              created by the model to %(metavar)s.\
                              The default value is 10000")

    parser.add_argument('-v', '--version', action='store_true',
                        dest='show_version',
                        help="Print the program version and exit")

    args = parser.parse_args()
    print("")

    if args.show_version:
        print("classtf - random forset classifier")
        print("version {0:d}.{1:d}.{2:d}".format(VERSION_MAJ,
                                                 VERSION_MIN,
                                                 VERSION_REV))
        print("")
        sys.exit(0)

    do_train = _check_file(args.train_filename)
    do_test = _check_file(args.test_filename)
    do_run = _check_file(args.run_filename)

    if args.do_feature_importance and (not do_test or not do_train):
        print(MSG_ERR_FIMPO)
        sys.exit(1)

    if not (do_run or do_test or do_train):
        parser.print_help()
        sys.exit(1)

    # Preliminary sanity checks
    if not args.model_dir:
        # if no model directory is specified hten create just
        # a temporary directoory
        tmp_dir = tempfile.TemporaryDirectory(prefix="rf-model-")
        model_dir = tmp_dir.name
    else:
        model_dir = args.model_dir
        # If we have specified the model directory but we are not
        # performing a training then we want to read the save we saved
        # early. In this case let's check if it really exists
        if not os.path.isdir(model_dir):
            if not args.train_filename:
                # no training...  bailing out
                print(MSG_ERR_DIR_NOT_FOUND.format(model_dir))
                sys.exit(1)
            elif not os.path.exists(model_dir):
                # the directory does not exist, let's try to create it
                try:
                    os.makedirs(model_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        # If somehow the direcotry has been created after the
                        # elif invocation, then just ignore the error, it
                        # exists that's what matters
                        print(MSG_ERR_DIR_PERM.format(model_dir))
                        sys.exit(1)
            else:
                print(MSG_ERR_DIR_NOT_FOUND.format(model_dir))
                sys.exit(1)
                # The path exists but is not a directory... bailing ou

    # NOTE: there is a bug in tensorflow version 1.12.0 that floods the
    #       console with warnings. They do not affect the execution, but are
    #       quite annoying. Let's hijack them to a log file...
    logfile = os.path.join(model_dir, 'log.txt')
    tf_logger = tf_logging._get_logger()
    tf_logger.removeHandler(tf_logger.handlers[0])
    tf_logger.addHandler(logging.FileHandler(logfile))

    train_timeout = timestr2sec(args.train_timeout)

    # Loading actual data
    if args.run_filename:
        x_data, _ = readinput(args.run_filename, None, args.skip_list)

    #
    # Here the actual program starts
    # Let's create a default tensorflow session
    #
    with tf.Session() as sess:
        ap.conf.max_lines = -1
        ap.conf.max_width = -1

        rf = RFModel(args.num_trees,
                     args.max_nodes,
                     tf_session=sess,
                     loss_treshold=args.loss_treshold,
                     batch_size=args.batch_size,
                     train_timeout=train_timeout)

        print("")
        print(MSG_INFO_MODELDIR.format(model_dir))
        print("")
        if do_train:
            print("Reading train dataset...")
            x_train, y_train, x_names, y_names = readinput(
                args.train_filename,
                args.class_target,
                args.skip_list,
                make_shadows=args.do_feature_importance)

            t_start = time.time()
            c_t_start = time.ctime()
            rf.train(x_train, y_train, model_dir)
            c_t_end = time.ctime()
            delta_t = datetime.timedelta(seconds=time.time()-t_start)

            with open(args.train_filename+'-train_log.txt', 'w') as outf:
                outf.write("Training starts at {}\n".format(c_t_start))
                outf.write(LOG_MODEL_INFO.format(rf))
                outf.write("\nFeatures: {}\n".format(x_names))
                outf.write("\nTarget class: {}\n".format(y_names))
                outf.write("Training ends at {}\n".format(c_t_end))
                outf.write("Elapsed time: {}\n".format(delta_t))

            del x_train
            del y_train
            print("")

        if do_test:
            print("Reading test dataset...")
            x_test, y_test, x_names, y_names = readinput(
                args.test_filename,
                args.class_target,
                args.skip_list,
                make_shadows=args.do_feature_importance)

            t_start = time.time()
            c_t_start = time.ctime()
            pred, cm = rf.test(x_test, y_test, model_dir)
            c_t_end = time.ctime()
            delta_t = datetime.timedelta(seconds=time.time()-t_start)

            t = Table.read(args.test_filename)
            newcol_name = 'PRED_CLASS'
            while newcol_name in t.colnames:
                newcol_name = '_'+newcol_name

            # add the predicted classes as last column in the dataset
            newcol = Column(pred, name=newcol_name)
            t.add_column(newcol)

            # save the test output
            catname = os.path.splitext(args.test_filename)[0] + '-test_output.fits'
            t.write(catname, format='fits')
            del t

            print("\nConfusion matrix:")
            print(cm)

            base_scores = score(cm)
            base_f1 = np.mean(base_scores, axis=0)[-1]

            # write some info in the test log
            with open(args.test_filename+'-test_log.txt', 'w') as outf:
                outf.write("Test starts at {}\n".format(c_t_start))
                outf.write(LOG_MODEL_INFO.format(rf))
                outf.write("\nFeatures: {}\n".format(x_names))
                outf.write("\nTarget class: {}\n".format(y_names))
                outf.write('CONFUSION MATRIX\n')
                outf.write(str(cm).replace('[', ' ').replace(']', ''))
                print("\n         precision   recall   f1-score")
                outf.write("\n\n         precision   recall   f1-score\n")
                fmt_str = "class {0:d}:  {1: 8.3f} {2: 8.3f}   {3: 8.3f}"
                for stats in base_scores:
                    print(fmt_str.format(*stats))
                    outf.write(fmt_str.format(*stats)+'\n')
                print("")
                outf.write("Test ends at {}\n".format(c_t_end))
                outf.write("Elapsed time: {}\n".format(delta_t))

            if args.do_feature_importance:
                num_features = x_test.shape[-1]

                t_start = time.time()
                c_t_start = time.ctime()

                imp_table = Table(
                    names=['ID', 'FEATURE', 'MDA', 'Z-SCORE', 'IMPORTANCE'],
                    dtype=['uint8', 'S32', 'float32', 'float32', 'float32'])
                imp_table['ID'].format = 'd'
                imp_table['FEATURE'].format = '>s'
                imp_table['MDA'].format = ' 10.6f'
                imp_table['Z-SCORE'].format = ' 10.6f'
                imp_table['IMPORTANCE'].format = ' 10.6f'

                # Using the Boruta algorithm
                print("Feature importance analysis...")
                for i in range(num_features):
                    print("Feature {} of {}: ".format(i+1, num_features))
                    # backup the column data
                    orig_col = x_test[..., i].copy()
                    # shuffle the i-th column
                    np.random.shuffle(x_test[..., i])

                    # compute the new confusion_matrix
                    _, ith_cm = rf.test(x_test, y_test, model_dir)

                    # restore the column
                    x_test[..., i] = orig_col.copy()
                    del orig_col

                    # compute the average score
                    ith_scores = score(ith_cm)
                    m_f1 = np.mean(ith_scores, axis=0)[-1]
                    imp_table.add_row([i, x_names[i], base_f1 - m_f1, 0, 0])
                c_t_end = time.ctime()
                delta_t = datetime.timedelta(seconds=time.time()-t_start)

                # computing the Z-score
                imp_table['Z-SCORE'] = imp_table['MDA']
                imp_table['Z-SCORE'] -= imp_table['MDA'].mean()
                imp_table['Z-SCORE'] /= imp_table['MDA'].std()

                # Finding the Maximum Z Shadow Accuracy
                MSZA = max(
                    x['Z-SCORE'] for x in imp_table
                    if x['FEATURE'].startswith('__shadow_')
                )

                imp_table['IMPORTANCE'] = imp_table['Z-SCORE']/MSZA

                print("\nAnalysis results:")
                imp_table.sort('IMPORTANCE')
                imp_table.reverse()
                imp_table.pprint(align=['>', '>', '>', '>', '>'])

                with open(args.test_filename+'-fimportance.txt', 'w') as outf:
                    outf.write("FI starts at {}\n\n".format(c_t_start))
                    outf.write(str(imp_table))
                    outf.write("\n\nFI ends at {}\n".format(c_t_end))
                    outf.write("Elapsed time: {}\n".format(delta_t))
            del x_test
            del y_test
            print("")

        if do_run:
            print("Reading input dataset...")
            x_data, _, x_names, _ = readinput(
                args.run_filename,
                None,
                args.skip_list)
            pred = rf.run(x_data, model_dir)
            del x_data

            # Read the original file
            t = Table.read(args.run_filename)

            # Check if the name is already used and if this is the case
            # then use a different name
            newcol_name = 'PRED_CLASS'
            while newcol_name in t.colnames:
                newcol_name = '_'+newcol_name

            # add the predicted classes as last column in the dataset
            newcol = Column(pred, name=newcol_name)
            t.add_column(newcol)

            catname = os.path.splitext(args.run_filename)[0] + '-rfout.fits'
            t.write(catname, format='fits')


if __name__ == '__main__':
    _main()
