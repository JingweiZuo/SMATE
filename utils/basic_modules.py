import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation
import tensorflow.keras.layers as ll
import random as rd 
import numpy as np

hidden_dim = 128
num_layers = 3
module_name = 'gru'
GPU = True

def rnn_cell(module_name):
    '''

    :param module_name:
    :return:
    '''

    # GRU   # -> The hidden dimension here is the number of hidden state in each RNN cell, a RNN cell is n-dimensional vector where n is the length of MTS
    if (module_name == 'gru'):
        rnn_cell = ll.GRUCell(units=hidden_dim, activation="tanh")
    # LSTM
    elif (module_name == 'lstm'):
        rnn_cell = ll.LSTMCell(units=hidden_dim, activation="tanh")
    # LSTM Layer Normalization
    '''elif (module_name == 'lstmLN'):
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation="tanh")'''
    return rnn_cell

def spatial_dynamic_block(input, pool_size, d_prime):
    ''' Create a spatial_dynamic_block
    Args:
        input: input tensor (samples, L, input_dims)
        pool_size: pooling window for AvgPooling1D layer
        comr_factor: the compression faction for hidden dimensions
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = AveragePooling1D(pool_size=pool_size, strides=1, padding="same")(input)
    se = Dense(d_prime,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

def encoder_smate(in_shape, pool_step, d_prime, kernels = [8, 5, 3]):
    input_ = Input(shape=in_shape)  # input: (samples, L, input_dims)
    L = in_shape[0]
    
    # temporal axis
    if (module_name == 'gru'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.GRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.GRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        
    elif (module_name == 'lstm'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(out_t) # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.LSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.LSTM(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
                
    # 1D-CNN
    out_s = spatial_dynamic_block(input_, kernels[0], d_prime)
    out_s = Conv1D(128, kernels[0], padding='same', kernel_initializer='he_uniform')(input_)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s)
    
    out_s = spatial_dynamic_block(out_s, kernels[1], 8)
    out_s = Conv1D(256, kernels[1], padding='same', kernel_initializer='he_uniform')(out_s)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s)
    
    out_s = spatial_dynamic_block(out_s, kernels[2], 16)
    out_s = Conv1D(128, kernels[2], padding='same', kernel_initializer='he_uniform')(out_s)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s) # L * D  

    #reduce latent space dimension (t & s axis)
    out_t = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(out_t)
    
    out_s = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(out_s) # L' * D
    
    
    out = ll.Concatenate(axis=-1)([out_t, out_s]) # (samples, L', 128*4 + 128)
    out = Dense(128)(out)
    #out = Dense(128)(out_s)
    out = BatchNormalization()(out)
    out = ll.LeakyReLU()(out)
    out = Dense(128)(out) 
    out = BatchNormalization()(out) # (samples, L', 128)
    
    model = Model(inputs=input_, outputs=out)
    
    return model

def decoder_smate(encoder, timesteps, data_dim, pool_step):
    input_ = encoder.output  # input: (batch_size, timesteps, latent_dim)

    out = ll.UpSampling1D(size=pool_step)(input_) 

    # 1D-CNN for reconstructing the spatial information
    #cells = [rnn_cell(module_name) for _ in range(num_layers)]
    #out = ll.RNN(cells, return_sequences=True)(out)
    
    # temporal axis
    if (module_name == 'gru'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(out)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.GRU(hidden_dim, return_sequences=True)(out)
            for i in range(num_layers - 1):
                out_t = ll.GRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        
    elif (module_name == 'lstm'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(out)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(out_t) # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.LSTM(hidden_dim, return_sequences=True)(out)
            for i in range(num_layers - 1):
                out_t = ll.LSTM(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
    
    out = ll.Dense(data_dim, activation='sigmoid')(out_t)

    model = Model(encoder.input, out)
    return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = ll.Reshape((1, filters))(se)
    se = ll.Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = ll.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = ll.multiply([input, se])
    return se

def encoder_smate_se(in_shape, pool_step):
    input_ = Input(shape=in_shape)  # input: (samples, L, input_dims)
    L = in_shape[0]
    
    # temporal axis
    if (module_name == 'gru'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.GRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.GRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        
    elif (module_name == 'lstm'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(out_t) # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.LSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.LSTM(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
 
    # 1D-CNN
    out_s = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(input_)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s)
    out_s = squeeze_excite_block(out_s)
    
    out_s = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(out_s)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s)
    out_s = squeeze_excite_block(out_s)
    
    out_s = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(out_s)
    out_s = BatchNormalization()(out_s)
    out_s = Activation('relu')(out_s) # L * D  

    #reduce latent space dimension (t & s axis)
    out_t = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(out_t)
    
    out_s = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(out_s) # 
    
    out = ll.Concatenate(axis=-1)([out_t, out_s]) # 1 * 2D
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = ll.LeakyReLU()(out)
    out = Dense(128)(out) 
    out = BatchNormalization()(out) # (samples, L', 128)
    
    model = Model(inputs=input_, outputs=out)
    
    return model

def encoder_smate_rdp(in_shape, pool_step):
    input_ = Input(shape=in_shape)  # input: (samples, L, input_dims)
    L = in_shape[0]
    D = in_shape[1]
    # temporal axis
    if (module_name == 'gru'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNGRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.GRU(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.GRU(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
        
    elif (module_name == 'lstm'):
        if (tf.test.is_gpu_available()):
            out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.CuDNNLSTM(hidden_dim, return_sequences=True)(out_t) # output: (batch_size, timesteps, hidden_dim)
        else:
            out_t = ll.LSTM(hidden_dim, return_sequences=True)(input_)
            for i in range(num_layers - 1):
                out_t = ll.LSTM(hidden_dim, return_sequences=True)(out_t)  # output: (batch_size, timesteps, hidden_dim)
 
    # 1D-CNN
    group_size = int(D * 1.5 / 3)
    in_s = []
    for i in range(3):
        idx_list = rd.sample(range(0, D), group_size)
        idx_array = np.array(idx_list)
        in_s_i = input_[:, :, i: i+group_size]
        in_s.append(in_s_i)

    out_s_11 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(in_s[0])
    out_s_11 = BatchNormalization()(out_s_11)
    out_s_11 = Activation('relu')(out_s_11)
    
    out_s_12 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(in_s[1])
    out_s_12 = BatchNormalization()(out_s_12)
    out_s_12 = Activation('relu')(out_s_12)
    
    out_s_13 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(in_s[2])
    out_s_13 = BatchNormalization()(out_s_13)
    out_s_13 = Activation('relu')(out_s_13)
    
    out_s_1 = [out_s_11, out_s_12, out_s_13]
    #out_s = K.concatenate((out_s_11, out_s_12, out_s_13), axis=0)
    
    conv1D_2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')
    conv1D_3 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')
    batch_norm1 = BatchNormalization()
    batch_norm2 = BatchNormalization()
    activ_relu1 = Activation('relu')
    activ_relu2 = Activation('relu')
    
    s_outs = []
    for s in out_s_1:
        s_out = conv1D_2(s)
        s_out = batch_norm1(s_out)
        s_out = activ_relu1(s_out)
        
        s_out = conv1D_3(s_out)
        s_out = batch_norm2(s_out)
        s_out = activ_relu2(s_out)
        
        s_out = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(s_out) # L * D
        s_outs.append(s_out)
    out_s = ll.Concatenate(axis=-1)(s_outs) # L * 3D
    
    #reduce latent space dimension (t & s axis)
    out_t = AveragePooling1D(pool_size=pool_step, strides=None, padding='same')(out_t)
    
    out = ll.Concatenate(axis=-1)([out_t, out_s]) # 1 * 4D
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = ll.LeakyReLU()(out)
    out = Dense(128)(out) 
    out = BatchNormalization()(out) # (samples, L', 128)
    
    model = Model(inputs=input_, outputs=out)
    
    return model