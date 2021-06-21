import numpy as np
import random as rd
import tensorflow.keras.backend as K

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

# Min Max Normalizer
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

def split_dataset(X, Y, sup_ratio, strategy='RandomSplit'):
    '''
    Objective: by supervised ratio, split the samples for supervised and unsupervised training

    :param X: a 3-D array: Nbr_samples x L x D
    :param Y: an 1-D array
    :param sup_ratio: the ratio of supervised samples
    :return:
        "X_sup, Y_sup, X_unsup, Y_unsup, n_classes"
    '''

    X_list_label, Y_list_label = list(), list()
    X_list_unlabel, Y_list_unlabel = list(), list()

    # compute the number of samples with labels to take
    n_samples = int(sup_ratio * len(Y))
    classes, counts_cl = np.unique(Y, return_counts=True)
    n_classes = len(classes)
    
    if sup_ratio == 1:
        return X, Y, np.array([]), np.array([]), n_classes
    
    if strategy == 'RandomSplit':

        ## not equal-split between classes
        idx_label = rd.sample(range(0, X.shape[0]), n_samples)
        X_sup = X[idx_label]
        Y_sup = Y[idx_label]
        # put the rest of the samples in the class in unlabeled samples
        idx_unlabel = np.array(range(len(Y)))  # the total number of instance in the class
        idx_unlabel = np.ma.array(idx_unlabel, mask=False)
        for i in idx_label:
            idx_unlabel.mask[i] = True
        idx_unlabel = idx_unlabel.compressed()  # a list of index for unlabeled samples

        X_unsup = X[idx_unlabel]
        Y_unsup = Y[idx_unlabel]

        return X_sup, Y_sup, X_unsup, Y_unsup, n_classes

    else:
        ## equal-split between classes

        n_per_class = int(n_samples / n_classes)
        for c in classes:
            idx_c = np.where(Y == c)[0]  # take an array without 'dtype'
            # get all samples for this class
            X_c = [X[i] for i in idx_c]
            # choose random instances
            if (n_per_class > len(X_c)):
                idx_label = range(0, len(X_c))
            else:
                idx_label = rd.sample(range(0, len(X_c)), n_per_class)
            X_label = [X_c[i] for i in idx_label]
            X_list_label.extend(X_label)
            Y_list_label.extend([c] * len(idx_label))

            # put the rest of the samples in the class in unlabeled samples
            new_idx_c = np.array(range(len(idx_c)))  # the total number of instance in the class
            new_idx_c = np.ma.array(new_idx_c, mask=False)
            for i in idx_label:
                new_idx_c.mask[i] = True
            idx_unlabel = new_idx_c.compressed()  # a list of index for unlabeled samples

            X_unlabel = [X_c[i] for i in idx_unlabel]

            X_list_unlabel.extend(X_unlabel)
            Y_list_unlabel.extend([c] * len(idx_unlabel))

            X_sup = np.array(X_list_label)
            Y_sup = np.asarray(Y_list_label).flatten()

            X_unsup = np.array(X_list_unlabel)
            Y_unsup = np.asarray(Y_list_unlabel).flatten()

        return X_sup, Y_sup, X_unsup, Y_unsup, n_classes


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
    return K.sum(K.pow(x-y, 2), axis = 2) # n * m

