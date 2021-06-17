
from utils.generic_utils import *
import time, functools

NB_CLASS = 0
MAX_TIMESTEPS = 0
MAX_NB_VARIABLES = 0

def z_normalization(mts):
    M = len(mts[0, :])
    for i in range(M):
        mts_i = mts[:, i]
        mean = np.mean(mts_i)
        std = np.std(mts_i)
        mts_i = (mts_i - mean) / std
        mts[:, i] = mts_i
    return mts

def convert_mts(rep, dataset, z_normal = False):
    global NB_CLASS, MAX_NB_VARIABLES
    
    seq = np.genfromtxt(rep + dataset, delimiter=' ', dtype=str, encoding="utf8")
    
    ids, counts = np.unique(seq[:,0], return_counts=True)
    No = ids.shape[0]
    D = seq.shape[1] - 3
    arr = np.asarray((ids, counts)).T
    Max_Seq_Len = np.max(arr[:,1].astype(np.int))
    out_X = np.zeros((No, D, Max_Seq_Len))
    out_Y = np.zeros((No, ))

    classes = np.unique(seq[:,2])
    NB_CLASS = classes.shape[0]
    MAX_NB_VARIABLES = D
    
    for idx, id in enumerate(ids):
        seq_cpy = seq[seq[:,0] == id]
        l_seq = seq_cpy.shape[0]
        out_X[idx, :, :l_seq] = np.transpose(seq_cpy[:, 3:])
        out_Y[idx] = seq_cpy[0, 2] 
        if z_normal: 
            out_X[idx, :, :l_seq] = np.transpose(z_normalization(np.transpose(out_X[idx, :, :l_seq])))
        
    return out_X, out_Y

def load_datasets(rep, ds_train, ds_test, z_normal = False):
    global MAX_TIMESTEPS
    
    
    X_train, y_train = convert_mts(rep, ds_train, z_normal)
    X_test, y_test = convert_mts(rep, ds_test, z_normal)
    
    # Normalize labels to [0, n_class]
    Y_train = np.zeros(len(y_train))
    Y_test = np.zeros(len(y_test))
    
    classes, counts_cl = np.unique(y_train, return_counts=True)
    print("class list is " + str(classes))

    mapping_c_l = {}  # a mappling between classes and labels
    for idx, c in enumerate(list(classes)):
        mapping_c_l.update({c: idx})
    
    i = 0
    for c in y_train:  # get keras labels
        Y_train[i] = mapping_c_l[c]
        i = i + 1
    i = 0
    for c in y_test:  # get keras labels
        Y_test[i] = mapping_c_l[c]
        i = i + 1
    
    classes_norm, counts_cl_conv = np.unique(Y_train, return_counts=True)
    print("class list (norm) is " + str(classes_norm))
    
    if X_train.shape[-1] != X_test.shape[-1]:
        MAX_TIMESTEPS = min(X_train.shape[-1], X_test.shape[-1])
        X_train = X_train[:,:,:MAX_TIMESTEPS]
        X_test = X_test[:,:,:MAX_TIMESTEPS]
    return X_train, X_test, Y_train, Y_test


# split labeled and unlabeled dataset

def split_dataset_train(X, Y, sup_ratio, strategy='RandomSplit'):
    '''
    Objective: by supervised ratio, split the samples for supervised and unsupervised training

    :param X: a 3-D array: Nbr_samples x L x D
    :param X_s: a 5-D array: Nbr_samples x L x D x D x Chl
    :param masking: a 3-D array: Nbr_samples x L x 1
    :param Y: an 1-D array
    :param sup_ratio: the ratio of supervised samples
    :return:
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
    
def get_PolluScope_dataset(rep, ds_train, ds_test, sup_ratio, split_strategy='EqualSplit'):
    start = time.time()
    X_train, X_test, Y_train, Y_test = load_datasets(rep, ds_train, ds_test, z_normal = True)
    end_load = time.time()
    print('time cost for loading data  : ' + str(end_load - start))
    
    X_sup, Y_sup, X_unsup, Y_unsup, n_classes = split_dataset_train(X_train, Y_train, sup_ratio, split_strategy)  # split the training set into labeled and unlabed samples
    end_split = time.time()
    print('time cost for splitting data  : ' + str(end_split - end_load))
    
    dataset = {}
    dataset.update({'X_train': X_train})
    dataset.update({'X_test': X_test})
    dataset.update({'Y_train': Y_train})
    dataset.update({'Y_test': Y_test})
    dataset.update({'X_sup': X_sup})
    dataset.update({'Y_sup': Y_sup})
    dataset.update({'X_unsup': X_unsup})
    dataset.update({'Y_unsup': Y_unsup})
    dataset.update({'n_classes': n_classes})

    return dataset
