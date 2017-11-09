'''
    Original Version cloned form github
    https://gist.github.com/dansileshi/21b52113ce0ecb6c0f56d6f7534bbaca

'''
import tensorflow as tf
import numpy as np

def bias_variable(name, shape, init='zero'):
    if init == 'zero':
        return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer())
    elif init == 'const':
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))
    else:
        print("*** Unrecognized Bias Initializer ***")


def weight_variable(name, shape, init='xavier'):
    if init == 'xavier':
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.xavier_initializer())
    elif init == 'orthogonal':
        return tf.get_variable(name, shape, tf.float32, tf.orthogonal_initializer())
    elif init == 'truncated':   # truncated_normal
        return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    else:
        print("*** Unrecognized Weight Initializer ***")


def batch_normalization(is_training, inputT, scope):
    '''
        tf.contrib.layer batch normalization document
        https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    '''
    def train_bn(inputT):
    	return tf.contrib.layers.batch_norm(inputT, is_training=True,
                center=False, updates_collections=None, scope=scope+"_bn")
    def test_bn(inputT):
    	return tf.contrib.layers.batch_norm(inputT, is_training=False,
                center=False, updates_collections=None, scope=scope+"_bn", reuse=True)
    return tf.cond(is_training, lambda:train_bn(inputT), lambda:test_bn(inputT))
    #return inputT 


def dropout( name, 
             is_training, 
             prev_layer, 
             keep_prob=0.8, 
             noise_shape=None, 
             seed=None
            ):
   
    def train_dp(prev_layer): 
        with tf.variable_scope(name+'_dropout') as scope: 
            drop = tf.nn.dropout(x=prev_layer, 
                                 keep_prob=keep_prob, 
                                 noise_shape=None, 
                                 seed=None, 
                                 name=name)
        return drop
    def test_dp():
        return prev_layer
    return tf.cond(is_training, lambda:train_dp(prev_layer), test_dp)


def avgpool3d( name, 
               prev_layer, 
               ksize=3, 
               strides=2, 
               padding='SAME'
              ):
    with tf.variable_scope(name) as scope:
        strides = [1, strides, strides,strides, 1]
        kernel = [1, ksize, ksize, ksize, 1]

        pool = tf.nn.avg_pool3d(prev_layer, ksize=kernel, 
                strides=strides, padding=padding, name=name)

    return pool


def maxpool3d( name, 
               prev_layer, 
               ksize=3, 
               strides=2, 
               padding='SAME'
              ):
    with tf.variable_scope(name) as scope:
        strides = [1, strides, strides,strides, 1]
        kernel = [1, ksize, ksize, ksize, 1]

        pool = tf.nn.max_pool3d(prev_layer, ksize=kernel, 
                strides=strides, padding=padding, name=name)

    return pool


def conv3d( name, 
            is_training, 
            prev_layer, 
            out_dim, 
            ksize=3, 
            strides=1,
            batch_norm=True,
            act_fn=tf.nn.relu, 
            weight_init='xavier', 
            bias_init='const', 
            padding='SAME'
           ):

    channels = prev_layer.get_shape()[-1].value
    # [batch stride, horizontal stride, vertical stride, channel stride]
    # usually [1, h ,v, 1]
    strides = [1, strides, strides, strides, 1]
    kernel = [ksize, ksize, ksize, channels, out_dim]
    with tf.variable_scope(name) as scope:
        kernel = weight_variable('weights', kernel, weight_init)
        conv = tf.nn.conv3d(prev_layer, kernel, strides, padding=padding)
        biases = bias_variable('biases', [out_dim], bias_init)
        conv = tf.nn.bias_add(conv, biases)
    if batch_norm is True:
        conv = batch_normalization(is_training, conv, name)
    if act_fn is not None:
        conv = act_fn(conv, name=name)

    return conv


def fc( name, 
        is_training, 
        prev_layer, 
        out_dim, 
        dropout_keep_prob=1.0,
        batch_norm=True, 
        act_fn=tf.nn.relu, 
        weight_init='xavier', 
        bias_init='const' 
       ):

    with tf.variable_scope(name) as scope:
        in_dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, in_dim])
        weights = weight_variable('weights', [in_dim, out_dim], weight_init)
        biases = bias_variable('biases', [out_dim], bias_init)
        fc = tf.add(tf.matmul(prev_layer_flat, weights), biases)
    if batch_norm is True:
        #fc = batch_normalization(is_training, fc, scope.name)
        fc = batch_normalization(is_training, fc, name)
    if act_fn is not None:
        fc = act_fn(fc, name=name+"_act_fn")
    fc = dropout(name, is_training, fc, dropout_keep_prob)

    return fc


def map_pretrain(name, pretrain):
    init = {'weights':None, 'biases':None, 'moving_mean':None, 'moving_variance':None}
    print('Mapping Layer ', name, ' to pretrain weights.')
    for layer, param in pretrain.items():
        layer_name = layer.split('/')[1]  # conv1/conv2/fc1/.... or conv1_bn/fc1_bn
        layer_type = layer.split('/')[-1][:-2]  # weights or biases
        if name == layer_name:
            init[layer_type] = tf.constant_initializer(param, dtype=tf.float32)
            print('Weight Pretrain ', layer, 'is restored.')
        if name+'_bn' == layer_name:
            init[layer_type] = tf.constant_initializer(param, dtype=tf.float32)
            print('Batch Norm Pretrain ', layer, 'is restored.')
    print('Layer ', name, ' initializer:')
    for type, param in init.items():
        if param == None:
            init[type] = tf.contrib.layers.xavier_initializer()
            print('Not find Pretrain ', type, '. Using Xavier Initializer.')
    return init


def batch_normalization_pretrain(is_training, inputT, init, scope):
    '''
        tf.contrib.layer batch normalization document
        https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    '''
    def train_bn(inputT):
    	return tf.contrib.layers.batch_norm(inputT, is_training=True,
                center=False, updates_collections=None, scope=scope+"_bn", 
                param_initializers=init)
    def test_bn(inputT):
    	return tf.contrib.layers.batch_norm(inputT, is_training=False,
                center=False, updates_collections=None, scope=scope+"_bn", reuse=True)
    return tf.cond(is_training, lambda:train_bn(inputT), lambda:test_bn(inputT))


def conv3d_pretrain( 
            pretrain,
            name, 
            is_training, 
            prev_layer, 
            out_dim, 
            ksize=(3, 3, 3), 
            strides=1,
            batch_norm=True,
            act_fn=tf.nn.relu, 
            bias_init='const', 
            padding='SAME'
           ):
    init = map_pretrain(name, pretrain)
    channels = prev_layer.get_shape()[-1].value
    strides = [1, strides, strides, strides, 1]
    kernel = [ksize[0], ksize[1], ksize[2], channels, out_dim]
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', kernel, tf.float32, init['weights'])
        conv = tf.nn.conv3d(prev_layer, kernel, strides, padding=padding)
        biases = tf.get_variable('biases', [out_dim], tf.float32, init['biases'])
        conv = tf.nn.bias_add(conv, biases)
    if batch_norm is True:
        conv = batch_normalization_pretrain(is_training, conv, init, name)
    if act_fn is not None:
        conv = act_fn(conv, name=name)

    return conv


def fc_pretrain( 
        pretrain,
        name, 
        is_training, 
        prev_layer, 
        out_dim, 
        dropout_keep_prob=1.0,
        batch_norm=True, 
        act_fn=tf.nn.relu, 
        bias_init='const' 
       ):

    init = map_pretrain(name, pretrain)
    in_dim = np.prod(prev_layer.get_shape().as_list()[1:])
    with tf.variable_scope(name) as scope:
        prev_layer_flat = tf.reshape(prev_layer, [-1, in_dim])
        weights = tf.get_variable('weights', [in_dim, out_dim], tf.float32, init['weights'])
        biases = tf.get_variable('biases', [out_dim], tf.float32, init['biases'])
        fc = tf.add(tf.matmul(prev_layer_flat, weights), biases)
    if batch_norm is True:
        fc = batch_normalization_pretrain(is_training, fc, init, name)
    if act_fn is not None:
        fc = act_fn(fc, name=name)
    fc = dropout(name, is_training, fc, dropout_keep_prob)

    return fc



def voxnet(inputs,
           is_training,
           num_classes=40,
           dropout_keep_prob=0.8,
           weights = dict(),
           scope='voxnet',
           name='voxnet',
           reuse=False):
    # tf.nn.conv3d only support floating numbers
    with tf.variable_scope(scope, name, [inputs], reuse=reuse) as sc:
        # net: 62 * 31 * 31 * 2
        net = inputs
        
        # net: 62 * 31 * 31 * 2 -> 30 * 15 * 15 * nf1
        nf1 = 8
        net = conv3d_pretrain(weights, 'conv1', is_training, net, batch_norm=True, out_dim=nf1, ksize=(4, 3, 3), strides=2, padding='VALID')
        net = conv3d_pretrain(weights, 'conv1_1', is_training, net, batch_norm=True, out_dim=nf1, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv1_2', is_training, net, batch_norm=True, out_dim=nf1, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv1_3', is_training, net, batch_norm=True, out_dim=nf1, ksize=(1, 1, 1), padding='VALID')
        net = dropout('dropout1', is_training, net, dropout_keep_prob)

        # net: 30 * 15 * 15 * nf1 -> 14 * 7 * 7 * nf2
        nf2 = 16
        net = conv3d_pretrain(weights, 'conv2', is_training, net, batch_norm=True, out_dim=nf2, ksize=(4, 3, 3), strides=2, padding='VALID')
        net = conv3d_pretrain(weights, 'conv2_1', is_training, net, batch_norm=True, out_dim=nf2, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv2_2', is_training, net, batch_norm=True, out_dim=nf2, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv2_3', is_training, net, batch_norm=True, out_dim=nf2, ksize=(1, 1, 1), padding='VALID')
        net = dropout('dropout2', is_training, net, dropout_keep_prob)

        # net: 14 * 7 * 7 * nf2 -> 6 * 3 * 3 * nf3
        nf3 = 32
        net = conv3d_pretrain(weights, 'conv3', is_training, net, batch_norm=True, out_dim=nf3, ksize=(4, 3, 3), strides=2, padding='VALID')
        net = conv3d_pretrain(weights, 'conv3_1', is_training, net, batch_norm=True, out_dim=nf3, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv3_2', is_training, net, batch_norm=True, out_dim=nf3, ksize=(1, 1, 1), padding='VALID')
        net = conv3d_pretrain(weights, 'conv3_3', is_training, net, batch_norm=True, out_dim=nf3, ksize=(1, 1, 1), padding='VALID')
        net = dropout('dropout3', is_training, net, dropout_keep_prob)

        # net: 6 * 3 * 3 * nf3 -> 54 * nf3-> bf
        bf = 1024
        net = fc_pretrain(weights, 'fc4', is_training, net,
                          out_dim=bf, 
                          dropout_keep_prob=0.5,
                          batch_norm=False,
                          act_fn = tf.nn.relu)

        # net: bf -> num_classes * 62
        net = fc_pretrain(weights, 'fc5', is_training, net,
                          out_dim=num_classes * 62, 
                          dropout_keep_prob=1,
                          batch_norm=False,
                          act_fn=None)

        ret = tf.reshape(net, [-1, 62, num_classes])
        return ret