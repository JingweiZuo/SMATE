# Generic utils

import numpy as np
import random as rd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda

# Data Normalization for each dimension
def z_normalization(mts):
    M = len(mts[0, :])
    for i in range(M):
        mts_i = mts[:, i]
        mean = np.mean(mts_i)
        std = np.std(mts_i)
        mts_i = (mts_i - mean) / (std + 10**(-8))
        mts[:, i] = mts_i
    return mts

# %% Min Max Normalizer
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


def get_mapping_c_l(rep_train, meta_csv):
    '''
    Convert the classes in dataset into training labels in Keras

    class_array: an array of classes for samples in dataset
    '''

    meta = np.genfromtxt(rep_train + meta_csv, delimiter=',', dtype=str, encoding="utf8")
    No = len(meta)
    class_array = meta[:, 1]  # the 2nd column in meta_csv is an array of classes
    classes, counts_cl = np.unique(class_array, return_counts=True)
    print("class list is " + str(classes))

    mapping_c_l = {}  # a mappling between classes and labels
    for idx, c in enumerate(list(classes)):
        mapping_c_l.update({c: idx})
    return mapping_c_l


# split labeled and unlabeled dataset

def split_dataset(X, X_s, masking, Y, sup_ratio, strategy='RandomSplit'):
    '''
    Objective: by supervised ratio, split the samples for supervised and unsupervised training

    :param X: a 3-D array: Nbr_samples x L x D
    :param X_s: a 5-D array: Nbr_samples x L x D x D x Chl
    :param masking: a 3-D array: Nbr_samples x L x 1
    :param Y: an 1-D array
    :param sup_ratio: the ratio of supervised samples
    :return:
    '''

    X_list_label, X_s_list_label, masking_list_label, Y_list_label = list(), list(), list(), list()
    X_list_unlabel, X_s_list_unlabel, masking_list_unlabel, Y_list_unlabel = list(), list(), list(), list()

    # compute the number of samples with labels to take
    n_samples = int(sup_ratio * len(Y))
    classes, counts_cl = np.unique(Y, return_counts=True)
    n_classes = len(classes)
    
    if sup_ratio == 1:
        return X, X, masking, Y, np.array([]), np.array([]), np.array([]), np.array([]), n_classes
    
    if strategy == 'RandomSplit':

        ## not equal-split between classes
        idx_label = rd.sample(range(0, X.shape[0]), n_samples)
        X_sup = X[idx_label]
        X_s_sup = X_s[idx_label]
        masking_sup = masking[idx_label]
        Y_sup = Y[idx_label]
        # put the rest of the samples in the class in unlabeled samples
        idx_unlabel = np.array(range(len(Y)))  # the total number of instance in the class
        idx_unlabel = np.ma.array(idx_unlabel, mask=False)
        for i in idx_label:
            idx_unlabel.mask[i] = True
        idx_unlabel = idx_unlabel.compressed()  # a list of index for unlabeled samples

        X_unsup = X[idx_unlabel]
        X_s_unsup = X_s[idx_unlabel]
        masking_unsup = masking[idx_unlabel]
        Y_unsup = Y[idx_unlabel]

        return X_sup, X_s_sup, masking_sup, Y_sup, X_unsup, X_s_unsup, masking_unsup, Y_unsup, n_classes

    else:
        ## equal-split between classes

        n_per_class = int(n_samples / n_classes)
        for c in classes:
            idx_c = np.where(Y == c)[0]  # take an array without 'dtype'
            # get all samples for this class
            X_c = [X[i] for i in idx_c]
            X_s_c = [X_s[i] for i in idx_c]
            masking_c = [masking[i] for i in idx_c]
            # choose random instances
            if (n_per_class > len(X_c)):
                idx_label = range(0, len(X_c))
            else:
                idx_label = rd.sample(range(0, len(X_c)), n_per_class)
            X_label = [X_c[i] for i in idx_label]
            X_s_label = [X_s_c[i] for i in idx_label]
            masking_label = [masking_c[i] for i in idx_label]
            X_list_label.extend(X_label)
            X_s_list_label.extend(X_s_label)
            masking_list_label.extend(masking_label)
            Y_list_label.extend([c] * len(idx_label))

            # put the rest of the samples in the class in unlabeled samples
            new_idx_c = np.array(range(len(idx_c)))  # the total number of instance in the class
            new_idx_c = np.ma.array(new_idx_c, mask=False)
            for i in idx_label:
                new_idx_c.mask[i] = True
            idx_unlabel = new_idx_c.compressed()  # a list of index for unlabeled samples

            X_unlabel = [X_c[i] for i in idx_unlabel]
            X_s_unlabel = [X_s_c[i] for i in idx_unlabel]
            masking_unlabel = [masking_c[i] for i in idx_unlabel]

            X_list_unlabel.extend(X_unlabel)
            X_s_list_unlabel.extend(X_s_unlabel)
            masking_list_unlabel.extend(masking_unlabel)
            Y_list_unlabel.extend([c] * len(idx_unlabel))

            X_sup = np.array(X_list_label)
            X_s_sup = np.array(X_s_list_label)
            masking_sup = np.array(masking_list_label)
            Y_sup = np.asarray(Y_list_label).flatten()

            X_unsup = np.array(X_list_unlabel)
            X_s_unsup = np.array(X_s_list_unlabel)
            masking_unsup = np.array(masking_list_unlabel)
            Y_unsup = np.asarray(Y_list_unlabel).flatten()

        return X_sup, X_s_sup, masking_sup, Y_sup, X_unsup, X_s_unsup, masking_unsup, Y_unsup, n_classes


# select real samples from dataset
def load_real_samples(X, Y, n_samples):
    '''

    :param X: a list of 2-D array for samples
    :param Y: an 1-D array of class labels
    :param n_samples:   the number of samples to be load
    :return:
    [X_samples, Y_samples]: the randomly selected samples and class labels
    y_virtual: Fake / Real label

    '''
    # choose random instances
    idx = rd.sample(range(0, len(X)), n_samples)
    X_samples = [X[i] for i in idx]  # X is a list of array
    Y_samples = Y[idx]  # Y should be an array

    # generate real/fake class labels
    y_virtual = np.ones((n_samples, 1))

    return [X_samples, Y_samples], y_virtual


# Compute the Correlation between dimensions
def mtx_correlation(X, channels):
    '''

    :param X:   Input list of 2-D array, N x L
    :param channels:    the channels at each time stamps
    :return:    A list of 4-D array, L x N x N x nbr_chl
    '''
    import time
    mtx_corr_list = list()
    print("total number of samples is " + str(len(X)))

    start = time.time()
    for idx, x in enumerate(X):
        L = x.shape[0]  # the length of MTS
        N = x.shape[1]  # the number of MTS dimension
        mtx_corr = np.zeros((L, N, N, len(channels)))

        for l in range(L):
            # for each time stamp in MTS
            mtx_corr_l = np.zeros((N, N, len(channels)))
            for c_idx, c in enumerate(channels):
                # c is the size of MTS segment
                # each channel, output a
                # keep the same length of each correlation sequence
                if (l < c):  # for the first MTS segments
                    mts_seg = x[:l, :]
                else:
                    mts_seg = x[l - c:l, :]  # split a MTS segment, mts_seg.shape = c x N
                mts_ij = np.corrcoef(np.transpose(mts_seg[:, :]))  # input N x c matrice, output N x N matrice,
                mts_ij = np.nan_to_num(mts_ij, nan=0)  # may have 'nan' value in the  matrice as the segment may be plat

                mtx_corr_l[:, :, c_idx] = mts_ij
            mtx_corr[l, :, :, :] = mtx_corr_l
        mtx_corr_list.append(mtx_corr)
        if idx % 100 == 0:
            print('time cost until round ' + str(idx) + ' is ' + str(time.time() - start))
    return mtx_corr_list


# Get the max length of MTS samples
def get_max_seq_len(X):
    '''

    :param X: A list of MTS samples (2-D array: length x dim)
    :return: the max length of MTS samples
    '''
    No = len(X)
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, X[i].shape[0])
    return Max_Seq_Len


# Padding the MTS batch into identical length (i.g., Max_Seq_Len)
def padding_variable_length(X_samples, Max_Seq_Len):
    '''

    :param X_samples: a batch/list of samples 2-D array: length x dim
    :return:
        - Xpad: A 3-D array of No x length x dim
        - L_samples: a list of length of initial samples

    '''
    print('total number of samples is ' + str(len(X_samples)))
    No = len(X_samples)
    dimension = len(X_samples[0][0, :])
    L_samples = list()
    for i in range(No):
        L_samples.append(X_samples[i].shape[0])
    # Padding and Masking
    special_value = 0
    Xpad = np.zeros((No, Max_Seq_Len, dimension))

    for s, x in enumerate(X_samples):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x

    return Xpad, L_samples


# Padding the Correlation matrix into identical length
def padding_corr_matrix(mtx_corr_list, Max_Seq_Len):
    '''

    :param mtx_corr_list: List of 4-D array "L  x N x N x Chl"
    :return:
        - mtx_corr_pad: A 5-D array "No x L x N x N x Chl  "

    '''
    No = len(mtx_corr_list)
    Chl = mtx_corr_list[0].shape[-1]
    dimension = mtx_corr_list[0].shape[1]

    # Padding and Masking
    mtx_corr_pad = np.zeros((No, Max_Seq_Len, dimension, dimension, Chl))
    for s, x in enumerate(mtx_corr_list):
        seq_len = x.shape[0]
        mtx_corr_pad[s, 0:seq_len, :, :, :] = x
    return mtx_corr_pad


# get the (padded) temporal and spatia samples, as well as masking array
def generate_real_samples(X, X_s, masking, Y, n_samples):
    # choose random instances
    if (n_samples >= X.shape[0]):
        n_samples = X.shape[0]
    idx = rd.sample(range(0, len(X)), n_samples)
    # X_samples = [X[i] for i in idx] #X is a list of array N_samples * 'L x D Chl'
    # X_s_samples = [X_s[i] for i in idx] #X_s is a list of array N_samples * 'L x D x D x Chl'
    X_samples = X[idx]  # X is an array of N_samples * 'L x D Chl'
    X_s_samples = X_s[idx]  # X_s is an array of N_samples * 'L x D x D x Chl'

    masking_samples = masking[idx]  # masking should be an array of 'N_samples x L x 1'
    Y_samples = Y[idx]  # Y should be an array of 'N_samples'

    # generate real/fake class labels
    # real sample: -1
    # fake sample: 1
    y_virtual = np.ones((n_samples, 1))

    return [X_samples, X_s_samples, masking_samples, Y_samples], y_virtual


# generate random arrays with predefined data dimensions
def random_generator(batch_size, data_dim, masking_seq, Max_Seq_Len):
    '''
        Create a 3-D array where each sample has different length, the extra parts are filled by 0

    :param batch_size: number of noise samples
    :param data_dim: the dimension number of each sample
    :param masking_seq: the mask sequence to mark the length of the sequences
    :param Max_Seq_Len: the max sequence length
    :return:   A 3-D array
    '''

    Zs = np.random.uniform(0., 1, [batch_size, Max_Seq_Len, data_dim])
    Zs = np.multiply(Zs, masking_seq)
    '''
    Zs = np.zeros([batch_size, Max_Seq_Len, z_dim])
    for i in range(batch_size):
        Z = np.random.uniform(0., 1, [Max_Seq_Len, data_dim])

        Zs[i, :L[i], :] = Z'''

    return Zs


# generate fake representation from random noise
def generate_fake_reprs(generator, batch_size, data_dim, L, masking_seq):
    # generate MTS points in latent space
    Z = random_generator(batch_size, data_dim, L)
    # predict outputs
    H_fake = generator.predict([Z, masking_seq])
    # create class labels
    Y_fake = np.zeros((batch_size, 1))
    return H_fake, Y_fake

# euclidean distance between two arrays
def euclidean_dist(x, y):
    # x: n * d
    # y: m * d
    n = x.shape[0]
    d = x.shape[1]
    m = y.shape[0]
    
    assert d == y.shape[1]

    x = K.repeat(x, m) # n * m * d
    y = K.expand_dims(y, axis=0) # 1 * m * d
    #y = Lambda(lambda t: K.gather(t, [0] * n))(y)
    return K.sum(K.pow(x-y, 2), axis = 2) # n * m

def euclidean_dist_mts(x, y):
    # x: n * L * d
    # y: m * L * d
    n = x.shape[0]
    l = x.shape[1]
    d = x.shape[2]
    m = y.shape[0]
    
    assert d == y.shape[2]
    
    x = K.reshape(x, shape=(n, l*d))
    y = K.reshape(y, shape=(m, l*d))

    x = K.repeat(x, m) # n * m * d'
    y = K.expand_dims(y, axis=0) # 1 * m * d'
    #y = Lambda(lambda t: K.gather(t, [0] * n))(y)
    return K.sum(K.pow(x-y, 2), axis = 2) # n * m

