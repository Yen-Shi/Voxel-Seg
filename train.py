import os
import tensorflow as tf
import numpy as np
import argparse
import sys
import time

import provider as pv
import model

FLAGS = None

def random_shuffle(x):
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    x = x[perm]
    return x, perm

def main(_):
    learningRate = FLAGS.learningRate
    epoch_step   = FLAGS.epoch_step
    num_classes  = FLAGS.orig_num_classes
    batchSize    = FLAGS.batchSize
    i_momentum   = FLAGS.momentum

    # Read class_hist
    classes_hists   = pv.readClassesHist(FLAGS.class_hist_file, num_classes)

    # set SGD optimizer's configuration
    decay_steps   = 1
    decay_rate    = FLAGS.learningRateDecay
    
    # set logger's configuration

    with tf.name_scope('input_variables'):
        x           = tf.placeholder(tf.float32, [None, 62, 31, 31, 2], name="x_input")
        ori_y       = tf.placeholder(tf.int32, [None, 1, 62])
        c_weights   = tf.placeholder(tf.float32, [num_classes], name="class_weights")
        is_training = tf.placeholder(tf.bool, name="is_training")

    res_y = tf.reshape(ori_y, [-1, 62])
    y_    = tf.one_hot(res_y, num_classes, name="y_true") # y_: 64 x 62 x 42
    y     = model.voxnet(x, is_training, num_classes)

    with tf.name_scope('weighted_loss'):
        # Ref: https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
        y1D             = tf.reshape(y, [-1, num_classes])
        y_1D            = tf.reshape(y_, [-1, num_classes])
        weights         = tf.reduce_sum(y_1D * c_weights, axis=1)
        ori_losses      = tf.nn.softmax_cross_entropy_with_logits(labels=y_1D, logits=y1D)
        loss            = tf.reduce_mean(ori_losses * weights)

    with tf.name_scope('optimizer'):
        # Ref: https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
        #      https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        #      https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
        #      http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
        global_step     = tf.Variable(0, trainable=False)
        starter_rate    = tf.Variable(0.01, trainable=False)
        learning_rate   = tf.train.exponential_decay(learning_rate=starter_rate,
                                                     global_step=global_step,
                                                     decay_steps=decay_steps,
                                                     decay_rate=decay_rate,
                                                     staircase=True)
        momentum        = tf.Variable(0.9, trainable=False)
        train_step = (
            tf.train.MomentumOptimizer(learning_rate, momentum)
            .minimize(loss=loss, global_step=global_step)
        )

    with tf.name_scope('num_of_correct'):
        mask = tf.reshape(tf.greater(c_weights, 0), [1, 1, num_classes])
        mask = tf.tile(mask, [tf.shape(y)[0], tf.shape(y)[1], 1])
        A = tf.where(mask, y, tf.zeros(tf.shape(y)))
        A = tf.cast(tf.argmax(A, axis=2), dtype=tf.int32)
        num_correct = tf.reduce_sum(tf.cast(tf.equal(A, res_y), dtype=tf.int32))

        B = tf.where(mask, y_, tf.zeros(tf.shape(y_)))
        B = tf.where(tf.equal(tf.reduce_sum(B, axis=2), 0),
                     tf.zeros([tf.shape(B)[0], 62], dtype=tf.int32),
                     tf.cast(tf.argmax(B, axis=2), dtype=tf.int32))
        num_total   = tf.reduce_sum(tf.cast(tf.logical_not(tf.logical_or(tf.equal(res_y, 0), tf.equal(res_y, num_classes-1))), dtype=tf.int32))
        num_choose  = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(B, 0)), dtype=tf.int32))

    answer = tf.argmax(y, axis=2)



    with tf.Session() as sess:
        saver = tf.train.Saver()
        graph_location = FLAGS.saveGraph
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location + '/train', sess.graph)
        train_writer.add_graph(tf.get_default_graph())
        test_writer = tf.summary.FileWriter(graph_location + '/test', sess.graph)
        test_writer.add_graph(tf.get_default_graph())
        # merged = tf.summary.merge_all()
        # tf.summary.scalar('accuracy', accuracy)
        # summary = sess.run(merged, feed_dict=feed_dict())
        # train_writer.add_summary(summary, i)

        # Load training and testing files
        train_files = pv.getDataFiles(FLAGS.train_data)
        test_files = pv.getDataFiles(FLAGS.test_data)
        print('#train_files = {}'.format(train_files.shape[0]))
        print('#test_files = {}'.format(test_files.shape[0]))

        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.max_epoch):
            # Start training
            start_time = time.time()

            num_of_files = train_files.shape[0]
            if epoch % epoch_step == 0 and epoch:
                learningRate = learningRate / 2

            train_files, _ = random_shuffle(train_files)

            print('Training epoch {}: |'.format(epoch+1), end='')
            total = 0    # annotated
            watched = 0  # in class hist
            correct = 0  # correct in class hist
            for fn in range(num_of_files):
                current_data, current_label = pv.loadDataFile(train_files[fn], num_classes)
                current_data, perm = random_shuffle(current_data)
                current_label = current_label[perm]
                current_data = current_data.swapaxes(1,2).swapaxes(2,3).swapaxes(3,4)

                num_of_batch = (current_data.shape[0] // batchSize)
                end_index    = num_of_batch * batchSize
                datas  = np.split(current_data[0:end_index], num_of_batch)
                labels = np.split(current_label[0:end_index], num_of_batch)
                
                for bn in range(num_of_batch):
                    feed_dict = {
                        x: datas[bn],
                        ori_y: labels[bn],
                        c_weights: classes_hists,
                        is_training: True,
                        starter_rate: learningRate,
                        momentum: i_momentum
                    }
                    # test = sess.run(, feed_dict=feed_dict)
                    # print(np.array(test).shape)
                    # print(test)
                    _, n_t, n_e, n_c = sess.run([train_step, num_total, num_choose, num_correct], feed_dict=feed_dict)
                    total += n_t
                    watched += n_e
                    correct += n_c
                    # import collections as ccc
                    # print('Answers: ', ccc.Counter(labels[bn].ravel()))
                    # print('Predict answers: ', ccc.Counter(ans.ravel()))
                    # print('Correct: {} | {} | {}'.format(n_t, n_e, n_c))
                print('=', end="")
                sys.stdout.flush()
            print('|')
            elapsed_time = time.time() - start_time
            print('Training epoch {}, time: {}'.format(epoch+1, elapsed_time))
            print('Training accuracy: {} | {} | {} , {}%'.format(total, watched, correct, correct / watched * 100))

#################################################################################################################

            # Start testing
            start_time   = time.time()
            num_of_files = test_files.shape[0]
            testBatch    = 100

            print('Testing epoch {}: |'.format(epoch+1), end='')
            total = 0    # annotated
            watched = 0  # in class hist
            correct = 0  # correct in class hist
            for fn in range(num_of_files):
                current_data, current_label = pv.loadDataFile(test_files[fn], num_classes)
                current_data = current_data.swapaxes(1,2).swapaxes(2,3).swapaxes(3,4)

                num_of_batch = (current_data.shape[0] // testBatch)
                end_index    = num_of_batch * testBatch
                datas  = np.split(current_data[0:end_index], num_of_batch)
                labels = np.split(current_label[0:end_index], num_of_batch)
                
                for bn in range(num_of_batch):
                    feed_dict = {
                        x: datas[bn],
                        ori_y: labels[bn],
                        c_weights: classes_hists,
                        is_training: False,
                    }
                    n_t, n_e, n_c = sess.run([num_total, num_choose, num_correct], feed_dict=feed_dict)
                    total += n_t
                    watched += n_e
                    correct += n_c
                print('=', end="")
                sys.stdout.flush()
            print('|')
            elapsed_time = time.time() - start_time
            print('Testing epoch {}, time: {}'.format(epoch+1, elapsed_time))
            print('Testing accuracy: {} | {} | {} , {}%'.format(total, watched, correct, correct / watched * 100))
            if epoch % 5 == 0:
                print('Save model for every 5 epochs !')
                model_location = FLAGS.saveModel
                save_path = saver.save(sess, model_location + '/voxnet-model.ckpt')
                print("Model saved in file: %s" % save_path)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Register commands
    # parser.add_argument('-s', '--save', type=str,
    #                     default='./logs',
    #                     help='Subdirectory to save logs')
    parser.add_argument('--saveGraph', type=str,
                        default='./graph',
                        help='Subdirectory to save graph')
    parser.add_argument('--saveModel', type=str,
                        default='./model',
                        help='Subdirectory to save model')
    parser.add_argument('-b', '--batchSize', type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('-r', '--learningRate', type=float,
                        default=0.01,
                        help='Learning rate')
    parser.add_argument('--learningRateDecay', type=float,
                        default=1e-7,
                        help='Learning rate decay')
    # Weight decay has not been used.
    parser.add_argument('--weigthDecay', type=float,
                        default=0.0005,
                        help='Weight decay')
    parser.add_argument('-m','--momentum', type=float,
                        default=0.9,
                        help='Mementum')
    parser.add_argument('--epoch_step', type=int,
                        default=20,
                        help='Epoch step')
    parser.add_argument('-g', '--gpu_index', type=int,
                        default=0,
                        help='GPU index (start from 0)')
    parser.add_argument('--max_epoch', type=int,
                        default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--train_data', type=str,
                        default='trainval_shape_voxel_data_list.txt',
                        help='Txt file containing train h5 filenames')
    parser.add_argument('--test_data', type=str,
                        default='test_shape_voxel_data_list.txt',
                        help='Txt file containing test h5 filenames')
    parser.add_argument('--retrain', type=str,
                        default='',
                        help='Retrain model')
    parser.add_argument('--class_hist_file', type=str,
                        default='classes_hist.txt',
                        help='Histogram for weight norm')    
    # parser.add_argument('--class_map_file', type=str,
    #                     default='',
    #                     help='')    
    parser.add_argument('--orig_num_classes', type=int,
                        default='42',
                        help='')
    # Have not used
    # --jitter_step    (default 2)    jitter augmentation step size
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)