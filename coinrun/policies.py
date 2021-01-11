import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

from coinrun.km_distributions import km_make_pdtype

from coinrun.config import Config

def impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out
    
    out = images
    inter_features = []
    attention_map = []
    for depth in depths:
        out = conv_sequence(out, depth)
        if depth == 16:
            attention_map = tf.math.reduce_mean(out, axis=3, keepdims=True)
            att_spatial_dim = attention_map.shape[1]
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], -1])
            attention_map = tf.nn.softmax(attention_map, axis=1)
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], att_spatial_dim, att_spatial_dim, 1])
            attention_map = tf.tile(attention_map, [1, 1, 1, depth])
        inter_features.append(out)
        
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, inter_features, attention_map, dropout_assign_ops

def soft_att_impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    
    # Attention networks
    count, att_index = 0, 0
    for depth in depths:
        out = conv_sequence(out, depth)
        if count == att_index:
            attention_net = out
            att_dim = depths[count]
            attention_map = tf.math.reduce_mean(out, axis=3, keepdims=True)
            att_spatial_dim = attention_map.shape[1]
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], -1])
            attention_map = tf.nn.softmax(attention_map, axis=1)
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], att_spatial_dim, att_spatial_dim, 1])
            attention_map = tf.tile(attention_map, [1, 1, 1, att_dim])
            out = attention_map * out
        count += 1
            
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, attention_map, dropout_assign_ops


def random_impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))

    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img = tf.layers.conv2d(img, randcnn_depth, 3, padding='same', kernel_initializer=tf.initializers.glorot_normal(), trainable=False, name='randcnn')
    out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')
    
    attention_map = []
    for depth in depths:
        out = conv_sequence(out, depth)
        if depth == 16:
            attention_map = tf.math.reduce_mean(out, axis=3, keepdims=True)
            att_spatial_dim = attention_map.shape[1]
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], -1])
            attention_map = tf.nn.softmax(attention_map, axis=1)
            attention_map = tf.reshape(attention_map, [attention_map.shape[0], att_spatial_dim, att_spatial_dim, 1])
            attention_map = tf.tile(attention_map, [1, 1, 1, depth])
        
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, attention_map, dropout_assign_ops

def random_impala_cnn_mc(mc_flag, MC_tot, images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))
            
    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img_list = []
    for index in range(MC_tot):
        rand_img_list.append(tf.layers.conv2d(img, randcnn_depth, 3, padding='same', kernel_initializer=tf.initializers.glorot_normal(), trainable=False, name='randcnn_%d' % index))
    
    rand_img = tf.reshape(tf.gather(rand_img_list, mc_flag), shape=rand_img_list[0].shape)
    out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')
       
        
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, dropout_assign_ops

def random_att_impala_cnn(images, clean_flag=0, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))

    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img = tf.layers.conv2d(img, randcnn_depth, 3, padding='same',\
                                kernel_initializer=tf.initializers.glorot_normal(),\
                                trainable=False, name='randcnn')
    
    '''
    # version 1: attention at the input
    attention_map = tf.layers.conv2d(img, 1, 3, padding='same',\
                                     kernel_initializer=tf.initializers.glorot_normal(), name='attention_cnn')
    attention_map = tf.sigmoid(attention_map)
    attention_map = tf.tile(attention_map, [1, 1, 1, 3])
    attention_img = attention_map * img + (1-attention_map) * rand_img
    '''
    
    # version 2
    att_index = 0
    att_dim = depths[att_index]
    attention_net = conv_sequence(images, att_dim)
    attention_map = tf.math.reduce_mean(attention_net, axis=3, keepdims=True)
    attention_map = tf.sigmoid(attention_map)
    attention_map = tf.tile(attention_map, [1, 1, 1, att_dim])
        
    if clean_flag == 1:
        for depth in depths:
            out = conv_sequence(out, depth)
    else:
        out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')
        count = 0
        for depth in depths:
            out = conv_sequence(out, depth)
            if count == att_index:
                out = attention_map * attention_net + (1 - attention_map) * out
            count += 1
        
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, attention_map, dropout_assign_ops

def random_soft_att_impala_cnn(images, clean_flag=0, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))

    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img = tf.layers.conv2d(img, randcnn_depth, 3, padding='same',\
                                kernel_initializer=tf.initializers.glorot_normal(),\
                                trainable=False, name='randcnn')
    
    '''
    attention_map = tf.layers.conv2d(img, 1, 3, padding='same',\
                                     kernel_initializer=tf.initializers.glorot_normal(), name='attention_cnn')
    attention_map = tf.reshape(attention_map, [attention_map.shape[0], -1])
    attention_map = tf.nn.softmax(attention_map, axis=1)
    attention_map = tf.reshape(attention_map, [attention_map.shape[0], 64, 64, 1])
    attention_map = tf.tile(attention_map, [1, 1, 1, 3])
    
    attention_img = attention_map * img + (1-attention_map) * rand_img
    '''
    
    # version 2
    att_index = 0
    att_dim = depths[att_index]
    attention_net = conv_sequence(images, att_dim)
    attention_map = tf.math.reduce_mean(attention_net, axis=3, keepdims=True)
    att_spatial_dim = attention_map.shape[1]
    attention_map = tf.reshape(attention_map, [attention_map.shape[0], -1])
    attention_map = tf.nn.softmax(attention_map, axis=1)
    attention_map = tf.reshape(attention_map, [attention_map.shape[0], att_spatial_dim, att_spatial_dim, 1])
    attention_map = tf.tile(attention_map, [1, 1, 1, att_dim])
        
    if clean_flag == 1:
        for depth in depths:
            out = conv_sequence(out, depth)
    else:
        out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')
        count = 0
        for depth in depths:
            out = conv_sequence(out, depth)
            if count == att_index:
                out = attention_map * attention_net + (1 - attention_map) * out
            count += 1
            
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, attention_map, dropout_assign_ops

def random_meta_impala_cnn(images, rand_params, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1
    depths=[16, 32, 32]
    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1]
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))

    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img = tf.nn.conv2d(img, rand_params, strides=(1, 1, 1, 1), padding='SAME')
        
    out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')
    
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, dropout_assign_ops

def nature_cnn(scaled_images, **conv_kwargs):
    """
    Model used in the paper "Human-level control through deep reinforcement learning" 
    https://www.nature.com/articles/nature14236
    """

    def activ(curr):
        return tf.nn.relu(curr)
    inter_features = []
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    inter_features.append(h)
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    inter_features.append(h2)
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    inter_features.append(h3)
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))), inter_features

def choose_cnn(images):
    arch = Config.ARCHITECTURE
    scaled_images = tf.cast(images, tf.float32) / 255.
    dropout_assign_ops = []
    attention_map = []
    if arch == 'nature':
        out, inter_features = nature_cnn(scaled_images)
    elif arch == 'impala':
        out, inter_features, attention_map, dropout_assign_ops = impala_cnn(scaled_images)
    elif arch == 'impalalarge':
        out, inter_features, attention_map, dropout_assign_ops = impala_cnn(scaled_images, depths=[32, 64, 64, 64, 64])
    else:
        assert(False)

    return out, inter_features, attention_map, dropout_assign_ops

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            h, _, _, self.dropout_assign_ops = choose_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            h, inter_features, attention_map, self.dropout_assign_ops = choose_cnn(processed_x)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        softout = tf.nn.softmax(self.pi)
        
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def get_inter_features(ob, *_args, **_kwargs):
            return sess.run(inter_features, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        def get_attention_map(ob, *_args, **_kwargs):
            return sess.run(attention_map, {X:ob})
        
        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.get_inter_features = get_inter_features
        self.get_softmax = get_softmax
        self.get_attention_map = get_attention_map
        
class SoftATTCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, attention_map, self.dropout_assign_ops = soft_att_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        softout = tf.nn.softmax(self.pi)
                
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def get_attention_map(ob, *_args, **_kwargs):
            return sess.run(attention_map, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        self.X = X
        self.H = h
        self.vf = vf
        
        self.step = step
        self.value = value
        self.get_attention_map = get_attention_map
        self.get_softmax = get_softmax
        
class CnnResetPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            h, _, _, self.dropout_assign_ops = choose_cnn(processed_x)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        with tf.compat.v1.variable_scope("target_model", reuse=tf.compat.v1.AUTO_REUSE):
            target_h, _, _, _ = choose_cnn(processed_x)
            target_vf = fc(target_h, 'v', 1)[:,0]
            self.target_pd, self.target_pi = self.pdtype.pdfromlatent(target_h, init_scale=0.01)
            
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.softout = tf.nn.softmax(self.pi)
        target_a0 = self.target_pd.sample()
        target_neglogp0 = self.pd.neglogp(target_a0)
        self.target_softout = tf.nn.softmax(self.target_pi)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def target_step(ob, *_args, **_kwargs):
            t_a, t_v, t_neglogp = sess.run([target_a0, target_vf, target_neglogp0], {X:ob})
            return t_a, t_v, self.initial_state, t_neglogp

        def target_value(ob, *_args, **_kwargs):
            return sess.run(target_vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.target_vf = target_vf
        self.target_step = target_step
        self.target_value = target_value
        
class CnnDQN(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            h, _, _, self.dropout_assign_ops = choose_cnn(processed_x)
            q_value = fc(h, 'dqn', ac_space.n)
            #q_value = tf.nn.relu(q_value)
            
        a0 = tf.argmax(q_value, axis=1)
        max_q_value = tf.reduce_max(q_value, axis=1)
        
        with tf.compat.v1.variable_scope("target_model", reuse=tf.compat.v1.AUTO_REUSE):
            target_h, _, _, _ = choose_cnn(processed_x)
            target_q_value = fc(target_h, 'dqn', ac_space.n)
            #target_q_value = tf.nn.relu(target_q_value)

        max_target_q_value = tf.reduce_max(target_q_value, axis=1)
        
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, qv = sess.run([a0, max_target_q_value], {X:ob})
            return a, qv, self.initial_state, _

        def target_q_val(ob, *_args, **_kwargs):
            return sess.run(max_target_q_value, {X:ob})

        self.X = X
        self.q_value = q_value
        
        self.step = step
        self.target_q_val = target_q_val

class RandomCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        mc_index = tf.placeholder(tf.int64, shape=[1], name='mc_index')
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, attention_map, self.dropout_assign_ops = random_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            clean_h, _, _, _ = impala_cnn(scaled_images)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        # for MC test
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            mc_h, _ = random_impala_cnn_mc(mc_index, Config.MC_ITER, scaled_images)
            mc_vf = fc(mc_h, 'v', 1)[:,0]
            self.mc_pd, self.mc_pi = self.pdtype.pdfromlatent(mc_h, init_scale=0.01)
                
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        softout = tf.nn.softmax(self.pi)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        mc_softout = tf.nn.softmax(self.mc_pi)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
        
        def get_attention_map(ob, *_args, **_kwargs):
            return sess.run(attention_map, {X:ob})
        
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        def get_mc_softmax(mc_index_in, ob, *_args, **_kwargs):
            return sess.run(mc_softout, {X:ob, mc_index:mc_index_in})
        
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf =clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean
        self.get_attention_map = get_attention_map
        self.get_softmax = get_softmax
        self.get_mc_softmax = get_mc_softmax
                
class RandomATTCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        mc_index = tf.placeholder(tf.int64, shape=[1], name='mc_index')
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, attention_map, self.dropout_assign_ops = random_att_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            clean_h, _, _ = random_att_impala_cnn(scaled_images, clean_flag=1)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        softout = tf.nn.softmax(self.pi)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
        
        def get_attention_map(ob, *_args, **_kwargs):
            return sess.run(attention_map, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf = clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean
        self.get_attention_map = get_attention_map
        self.get_softmax = get_softmax
        
class RandomSoftATTCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        mc_index = tf.placeholder(tf.int64, shape=[1], name='mc_index')
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, attention_map, self.dropout_assign_ops = random_soft_att_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            clean_h, _, _ = random_soft_att_impala_cnn(scaled_images, clean_flag=1)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        softout = tf.nn.softmax(self.pi)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
        
        def get_attention_map(ob, *_args, **_kwargs):
            return sess.run(attention_map, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf = clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean
        self.get_attention_map = get_attention_map
        self.get_softmax = get_softmax

class RandomMetaCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        num_action, meta_batch = 7, 1
        
        # setup for meta-learning
        nlstm = Config.META_NLSTM
        
        # inputs for meta-learner
        meta_pre_X, meta_pre_processed_x = observation_input(ob_space, Config.META_WINDOW * meta_batch)
        meta_pre_scaled_images = tf.cast(meta_pre_processed_x, tf.float32) / 255.
        meta_pre_act = tf.placeholder(tf.int64, shape=[Config.META_WINDOW])
        meta_pre_one_hot = tf.one_hot(meta_pre_act, 7)
        meta_pre_M = tf.placeholder(tf.float32, [Config.META_WINDOW * meta_batch]) #mask (done t-1)
        meta_pre_S = tf.placeholder(tf.float32, [meta_batch, nlstm*2]) #states
        
        # test samples at meta-training
        meta_post_X, meta_post_processed_x = observation_input(ob_space, Config.META_WINDOW * meta_batch)
        meta_post_scaled_images = tf.cast(meta_post_processed_x, tf.float32) / 255.
        
        # test sample at inference time
        meta_post_infer_X, meta_post_infer_processed_x = observation_input(ob_space, 1)
        meta_post_infer_scaled_images = tf.cast(meta_post_infer_processed_x, tf.float32) / 255.
        
        # random_params
        fixed_random_params = tf.placeholder(tf.float32, [3, 3, 3, 3])
        
        # meta-learner outputs
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            meta_pre_h, _, _ = random_impala_cnn(meta_pre_scaled_images)
            meta_pre_h = tf.stop_gradient(meta_pre_h)
            meta_act_em = tf.nn.relu(fc(meta_pre_one_hot, 'meta_lstm1_act', 16))
            
            xs = batch_to_seq(tf.concat([meta_pre_h, meta_act_em], 1), meta_batch, Config.META_WINDOW)
            ms = batch_to_seq(meta_pre_M, meta_batch, Config.META_WINDOW)

            h5, snew = lstm(xs, ms, meta_pre_S, 'meta_lstm1', nh=nlstm)
            meta_h_out = seq_to_batch(h5)
            meta_out = fc(meta_h_out, 'meta_lstm1_fc', 81)
            meta_out = tf.reshape(meta_out[-1], shape=(3, 3, 3, 3))
    
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            meta_h, _ = random_meta_impala_cnn(meta_post_scaled_images, fixed_random_params + meta_out)
            meta_vf = fc(meta_h, 'v', 1)[:,0]
            self.meta_pd, self.meta_pi = self.pdtype.pdfromlatent(meta_h, init_scale=0.01)

        # inference
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            meta_pre_infer_h, _ = random_meta_impala_cnn(meta_pre_scaled_images, fixed_random_params)
            meta_pre_act_em = tf.nn.relu(fc(meta_pre_one_hot, 'meta_lstm1_act', 16))
            xs_pre_infer = batch_to_seq(tf.concat([meta_pre_infer_h, meta_pre_act_em], 1), meta_batch, Config.META_WINDOW)
            ms_pre_infer = batch_to_seq(meta_pre_M, meta_batch, Config.META_WINDOW)
            h5_pre_infer, snew_pre_infer = lstm(xs_pre_infer, ms_pre_infer, meta_pre_S, 'meta_lstm1', nh=nlstm)
            meta_pre_h_out = seq_to_batch(h5_pre_infer)
            meta_pre_out = fc(meta_pre_h_out, 'meta_lstm1_fc', 81)
            meta_pre_out = tf.reshape(meta_pre_out[-1], shape=(3, 3, 3, 3))
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            meta_post_infer_h, _ = random_meta_impala_cnn(meta_post_infer_scaled_images, \
                                                                  fixed_random_params + meta_pre_out)
            self.meta_post_infer_pd, self.meta_post_infer_pi = self.pdtype.pdfromlatent(meta_post_infer_h, init_scale=0.01)
        
        meta_post_infer_a0 = self.meta_post_infer_pd.sample()
        meta_softout = tf.nn.softmax(self.meta_post_infer_pi)
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            post_infer_h, _ = random_meta_impala_cnn(meta_post_infer_scaled_images, fixed_random_params)
            self.post_infer_pd, self.post_infer_pi = self.pdtype.pdfromlatent(post_infer_h, init_scale=0.01)
        
        post_infer_a0 = self.post_infer_pd.sample()
        softout = tf.nn.softmax(self.post_infer_pi)
        
        # Original part
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, _, self.dropout_assign_ops = random_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            clean_h, _, _, _ = impala_cnn(scaled_images)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
            
        def meta_normal_step(ob, random_parm, *_args, **_kwargs):
            return sess.run(softout, {meta_post_infer_X: ob, fixed_random_params: random_parm})
        
        def meta_step(ob, prev, act, random_parm, pre_M, pre_S, *_args, **_kwargs):
            soft_out, new_para \
            = sess.run([meta_softout, meta_pre_out], {meta_pre_X: prev, meta_post_infer_X: ob, \
                                                   meta_pre_S: pre_S, meta_pre_M: pre_M, \
                                                   fixed_random_params: random_parm, meta_pre_act:act})
            return soft_out, new_para
        
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf = clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean        
        
        # meta-learner inputs
        self.meta_pre_X = meta_pre_X
        self.meta_post_X = meta_post_X
        self.meta_post_infer_X = meta_post_infer_X
        self.meta_pre_act = meta_pre_act
        self.meta_pre_M = meta_pre_M
        self.meta_pre_S = meta_pre_S
        
        # for training 
        self.meta_H = meta_h
        self.meta_vf = meta_vf
        self.meta_out = meta_out
        self.fixed_random_params = fixed_random_params
        
        # for meta-inference
        self.meta_normal_step = meta_normal_step
        self.meta_step = meta_step
        
class RandomPolicyMetaCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = km_make_pdtype(ac_space)
        num_action, meta_batch = 7, 1
        
        # setup for meta-learning
        nlstm = Config.META_NLSTM
        
        # inputs for meta-learner
        meta_pre_X, meta_pre_processed_x = observation_input(ob_space, Config.META_WINDOW * meta_batch)
        meta_pre_scaled_images = tf.cast(meta_pre_processed_x, tf.float32) / 255.
        meta_pre_act = tf.placeholder(tf.int64, shape=[Config.META_WINDOW])
        meta_pre_one_hot = tf.one_hot(meta_pre_act, 7)
        meta_pre_M = tf.placeholder(tf.float32, [Config.META_WINDOW * meta_batch]) #mask (done t-1)
        meta_pre_S = tf.placeholder(tf.float32, [meta_batch, nlstm*2]) #states
        
        # test samples at meta-training
        meta_post_X, meta_post_processed_x = observation_input(ob_space, Config.META_WINDOW * meta_batch)
        meta_post_scaled_images = tf.cast(meta_post_processed_x, tf.float32) / 255.
        
        # test sample at inference time
        meta_post_infer_X, meta_post_infer_processed_x = observation_input(ob_space, 1)
        meta_post_infer_scaled_images = tf.cast(meta_post_infer_processed_x, tf.float32) / 255.
        
        # random_params
        fixed_kernel = tf.placeholder(tf.float32, [256, 7])
        fixed_bias = tf.placeholder(tf.float32, [7])
        
        # meta-learner outputs
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            meta_pre_h, _, _ = random_impala_cnn(meta_pre_scaled_images)
            meta_pre_h = tf.stop_gradient(meta_pre_h)
            meta_act_em = tf.nn.relu(fc(meta_pre_one_hot, 'meta_lstm1_act', 16))
            
            xs = batch_to_seq(tf.concat([meta_pre_h, meta_act_em], 1), meta_batch, Config.META_WINDOW)
            ms = batch_to_seq(meta_pre_M, meta_batch, Config.META_WINDOW)

            h5, snew = lstm(xs, ms, meta_pre_S, 'meta_lstm1', nh=nlstm)
            meta_h_out = seq_to_batch(h5)
            meta_out_kernel = fc(meta_h_out, 'meta_lstm1_fc_kernel', 256*7)
            meta_out_bias = fc(meta_h_out, 'meta_lstm1_fc_bias', 7)
            meta_out_kernel = tf.reshape(meta_out_kernel[-1], shape=(256, 7))
            meta_out_bias = tf.reshape(meta_out_bias[-1], shape=(7,))

        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            meta_h, _, _ = random_impala_cnn(meta_post_scaled_images)
            meta_vf = fc(meta_h, 'v', 1)[:,0]
            self.meta_pd, self.meta_pi = self.pdtype.pdfromlatent_external(meta_h, meta_out_kernel, meta_out_bias)
            
        # inference
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            meta_pre_infer_h, _, _ = random_impala_cnn(meta_pre_scaled_images)
            meta_pre_act_em = tf.nn.relu(fc(meta_pre_one_hot, 'meta_lstm1_act', 16))
            xs_pre_infer = batch_to_seq(tf.concat([meta_pre_infer_h, meta_pre_act_em], 1), meta_batch, Config.META_WINDOW)
            ms_pre_infer = batch_to_seq(meta_pre_M, meta_batch, Config.META_WINDOW)
            h5_pre_infer, snew_pre_infer = lstm(xs_pre_infer, ms_pre_infer, meta_pre_S, 'meta_lstm1', nh=nlstm)
            meta_pre_h_out = seq_to_batch(h5_pre_infer)
            meta_pre_out_kernel = fc(meta_pre_h_out, 'meta_lstm1_fc_kernel', 256*7)
            meta_pre_out_bias = fc(meta_pre_h_out, 'meta_lstm1_fc_bias', 7)
            meta_pre_out_kernel = tf.reshape(meta_pre_out_kernel[-1], shape=(256, 7))
            meta_pre_out_bias = tf.reshape(meta_pre_out_bias[-1], shape=(7,))
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            meta_post_infer_h, _, _ = random_impala_cnn(meta_post_infer_scaled_images)
            self.meta_post_infer_pd, self.meta_post_infer_pi \
            = self.pdtype.pdfromlatent_external(meta_post_infer_h, meta_pre_out_kernel, meta_pre_out_bias)
        
        meta_post_infer_a0 = self.meta_post_infer_pd.sample()
        meta_softout = tf.nn.softmax(self.meta_post_infer_pi)
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            post_infer_h, _, _ = random_impala_cnn(meta_post_infer_scaled_images)
            self.post_infer_pd, self.post_infer_pi = self.pdtype.pdfromlatent(post_infer_h)
        post_infer_a0 = self.post_infer_pd.sample()
        softout = tf.nn.softmax(self.post_infer_pi)
        
        # Original part
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):    
            h, _, self.dropout_assign_ops = random_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h)
            
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            clean_h, _, _, _ = impala_cnn(scaled_images)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
            
        def meta_normal_step(ob, *_args, **_kwargs):
            return sess.run(softout, {meta_post_infer_X: ob})
        
        def meta_step(ob, prev, act, fix_ker, fix_bias, pre_M, pre_S, *_args, **_kwargs):
            soft_out, new_para \
            = sess.run([meta_softout, meta_pre_out], {meta_pre_X: prev, meta_post_infer_X: ob, \
                                                   meta_pre_S: pre_S, meta_pre_M: pre_M, \
                                                   fixed_kernel: fix_ker, fixed_bias:fix_bias, meta_pre_act:act})
            return soft_out, new_para
        
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf = clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean        
        
        # meta-learner inputs
        self.meta_pre_X = meta_pre_X
        self.meta_post_X = meta_post_X
        self.meta_post_infer_X = meta_post_infer_X
        self.meta_pre_act = meta_pre_act
        self.meta_pre_M = meta_pre_M
        self.meta_pre_S = meta_pre_S
        
        # for training 
        self.meta_H = meta_h
        self.meta_vf = meta_vf
        self.fixed_kernel = fixed_kernel
        self.fixed_bias = fixed_bias
        
        # for meta-inference
        self.meta_normal_step = meta_normal_step
        self.meta_step = meta_step
        

class VAEPolicy(object):
    def __init__(self, sess, z_size, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        self.dropout_assign_ops = []
        
        # input definition
        ob_shape = (nbatch, z_size)
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            activ = tf.nn.relu
            h1 = activ(fc(X, "shared_fc1", nh=1024, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, "shared_fc2", nh=1024, init_scale=np.sqrt(2)))
            vf = fc(h2, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        softout = tf.nn.softmax(self.pi)
        
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.get_softmax = get_softmax


class VAELstmPolicy(object):

    def __init__(self, sess, z_size, ac_space, nbatch, nsteps, nlstm=256, **conv_kwargs):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        self.dropout_assign_ops = []
        # input definition
        ob_shape = (nbatch, z_size)
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            xs = batch_to_seq(X, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            activ = tf.nn.relu
            h = activ(fc(h5, "h", nh=512,  init_scale=np.sqrt(2)))
            hh = activ(fc(h, "hh", nh=512, init_scale=np.sqrt(2)))
            vf = fc(hh, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(hh)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value
        self.get_softmax = get_softmax


class VAEATTPolicy(object):
    def __init__(self, sess, z_size, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        self.dropout_assign_ops = []
        
        # input definition
        ob_shape = (nbatch, z_size)
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")
        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            activ = tf.nn.relu
            h1 = activ(fc(X, "shared_fc1", nh=512, init_scale=np.sqrt(2)))
            
            # attention 
            attention_map = tf.nn.softmax(h1, axis=1)
            h1 = attention_map * h1
                        
            h2 = activ(fc(h1, "shared_fc2", nh=512, init_scale=np.sqrt(2)))
            vf = fc(h2, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        softout = tf.nn.softmax(self.pi)
        
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def get_softmax(ob, *_args, **_kwargs):
            return sess.run(softout, {X:ob})
        
        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.get_softmax = get_softmax
                
def get_policy():
    use_lstm = Config.USE_LSTM
    
    if use_lstm == 1:
        policy = LstmPolicy
    elif use_lstm == 0:
        policy = CnnPolicy
    elif use_lstm == 2:
        policy = RandomCnnPolicy
    elif use_lstm == 3:
        policy = RandomATTCnnPolicy
    elif use_lstm == 4:
        policy = RandomSoftATTCnnPolicy
    elif use_lstm == 5:
        policy = RandomMetaCnnPolicy
    elif use_lstm == 6:
        policy = RandomPolicyMetaCnnPolicy
    elif use_lstm == 7:
        policy = SoftATTCnnPolicy
    elif use_lstm == 8:
        policy = VAEPolicy
    elif use_lstm == 9:
        policy = VAEATTPolicy 
    elif use_lstm == 10:
        policy = VAELstmPolicy
    elif use_lstm == 1081:
        policy = CnnResetPolicy
    elif use_lstm == 8425:
        policy = CnnDQN
    else:
        assert(False)

    return policy
