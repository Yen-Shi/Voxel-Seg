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

def calculate_acc(conf_matrix, class_hist, num_classes):
    total = 0
    correct = 0
    for i in range(num_classes):
        if class_hist[i] != 0:
            total += np.sum(conf_matrix[i])
            correct += conf_matrix[i][i]
    return total, correct

def check_data(data, label):
    import collections as ccc
    print('data: ')
    print(data.shape)
    print(ccc.Counter(data.ravel()))
    print('lable: ')
    print(label.shape)
    print(ccc.Counter(label.ravel()))

def main(_):
    learningRate = FLAGS.learningRate
    epoch_step   = FLAGS.epoch_step
    num_classes  = FLAGS.orig_num_classes
    batchSize    = FLAGS.batchSize
    i_momentum   = FLAGS.momentum

    # Read class_hist
    classes_hists = pv.readClassesHist(FLAGS.class_hist_file, num_classes)

    # set SGD optimizer's configuration
    decay_steps   = 1
    decay_rate    = FLAGS.learningRateDecay
    
    # set logger's configuration

    with tf.name_scope('input_variables'):
        x               = tf.placeholder(tf.float32, [None, 62, 31, 31, 2], name="x_input")
        ori_y           = tf.placeholder(tf.int32, [None, 1, 62])
        c_weights       = tf.placeholder(tf.float32, [num_classes], name="class_weights")
        is_training     = tf.placeholder(tf.bool, name="is_training")

    with tf.name_scope('main'):
        res_y           = tf.reshape(ori_y, [-1, 62])
        y_              = tf.one_hot(res_y, num_classes, name="y_true") # y_: 64 x 62 x 42
        y               = model.voxnet(x, is_training, num_classes)
        y_1D            = tf.reshape(y_, [-1, num_classes])
        y1D             = tf.reshape(y, [-1, num_classes])

    with tf.name_scope('weighted_loss'):
        # Ref: https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
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
        learning_rate   = tf.train.inverse_time_decay(learning_rate=starter_rate,
                                                      global_step=global_step,
                                                      decay_steps=decay_steps,
                                                      decay_rate=decay_rate)
        momentum        = tf.Variable(0.9, trainable=False)
        train_step = (
            tf.train.MomentumOptimizer(learning_rate, momentum)
            .minimize(loss=loss, global_step=global_step)
        )

    with tf.name_scope('confusion_matrix'):
        # Ref: https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
        # Compute a per-batch confusion
        batch_confusion = tf.confusion_matrix(labels=tf.argmax(y_1D, axis=1),
                                              predictions=tf.argmax(y1D, axis=1),
                                              num_classes=num_classes,
                                              name='batch_confusion')
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(tf.zeros([num_classes, num_classes], dtype=tf.int32),
                                name='confusion')
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign(confusion + batch_confusion)

    answer = tf.argmax(y1D, axis=1)


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
            sess.run(tf.variables_initializer([global_step, confusion], name='init_gStep_confmtx'))
            conf_matrix = []
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
                    _, conf_matrix = sess.run([train_step, confusion_update], feed_dict=feed_dict)
                print('=', end="")
                sys.stdout.flush()
            print('|')
            elapsed_time = time.time() - start_time
            print('Training epoch {}, time: {}'.format(epoch+1, elapsed_time))
            total, correct = calculate_acc(conf_matrix, classes_hists, num_classes)
            print('Training accuracy: {} | {} , {}%'.format(total, correct, correct / total * 100))

#################################################################################################################

            # Start testing
            start_time   = time.time()
            num_of_files = test_files.shape[0]
            testBatch    = 100

            print('Testing epoch {}: |'.format(epoch+1), end='')
            sess.run(tf.variables_initializer([confusion], name='init_confmtx'))
            conf_matrix = []
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
                        is_training: False,
                    }
                    conf_matrix = sess.run(confusion_update, feed_dict=feed_dict)
                print('=', end="")
                sys.stdout.flush()
            print('|')
            elapsed_time = time.time() - start_time
            print('Testing epoch {}, time: {}'.format(epoch+1, elapsed_time))
            total, correct = calculate_acc(conf_matrix, classes_hists, num_classes)
            print('Testing accuracy: {} | {}, {}%'.format(total, correct, correct / total * 100))
            
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