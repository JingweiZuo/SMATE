#Loading MTS data of UEA

from utils.generic_utils import *
import time, functools

'''=================================================== Prepare data import (MTS-UEA) ========================================================'''



'''================================= Prepare data import (MTS-UEA) =================================='''

def import_data_UEA(rep, meta_csv, mapping_c_l):
    '''

    :param rep: the repository of dataset files
    :param meta_csv: the meta-data of dataset
    :return:
        X_train: list of 2-D array, length * dimension
        Labels: An 1-D array of labels
    '''
    meta = np.genfromtxt(rep + meta_csv, delimiter=',', dtype=str, encoding="utf8")
    No = len(meta)
    Names = meta[:, 0]
    class_array = meta[:, 1]
    L = class_array.shape[0]
    Labels = np.zeros(L)

    i = 0
    for c in class_array:  # get keras labels
        Labels[i] = mapping_c_l[c]
        i = i + 1

    X_train = list()

    for k in range(No):
        dsName = Names[k]
        raw_data = np.genfromtxt(rep + dsName + '.csv', delimiter=',', encoding="utf8",
                                 filling_values=0)  # No header in raw files
        raw_data = raw_data.astype(np.float32)  # remove timestamp and the last row (label)
        data = z_normalization(raw_data)
        data = MinMaxScaler(data)

        '''
            Just for CharacterTrajectories, the end of MTS is all O
        
        L_seq = 0
        for idx in range(raw_data.shape[0]):
            if raw_data[idx].all() == 0:
                L_seq = idx
                break

        data[L_seq:] = np.zeros_like(data[L_seq:])'''

        # As the samples may not have the same length, then we put them into a list
        X_train.append(data)
    #print(Labels)
    return X_train, Labels  # X_train is a list of 2-D array, length*dimension

def prepare_data_UEA(rep, meta_csv, mapping_c_l, mode='load'):  # mode = 'load'/'save' samples' spatial correlation from/to disk
    # Load meta-data about the dataset

    X_list, Y = import_data_UEA(rep, meta_csv,
                                mapping_c_l)  # a list of 2-D array, length*dimension; an 1-D array of labels

    Max_Seq_Len = get_max_seq_len(X_list)
    X, L = padding_variable_length(X_list, Max_Seq_Len)  # Padding the samples into an identical length

    ds_size = X.shape[0]
    maskings = np.zeros((ds_size, Max_Seq_Len, 1))  # padding the spatial correlation matrix
    for idx in range(ds_size):
        l_seq = L[idx]
        maskings[idx, :l_seq, :] = np.ones((l_seq, 1))

    start = time.time()  # counting the time of computing spatial correlation

    X_s = X
    # print(x_s[0, 20, :, :, 0]) # retrieve the 1st sample: mtx_corrs[0]

    end_corr = time.time()
    #print('time cost for computing spatial loading/correlation : ' + str(end_corr - start))
    
    return X, X_s, maskings, Y

def get_UEA_dataset(rep_ds_train, rep_ds_test, meta_csv, sup_ratio, mode = 'load', split_strategy='EqualSplit'):
    start = time.time()
    mapping_c_l = get_mapping_c_l(rep_ds_train, meta_csv)
    X_train, X_s_train, masking_train, Y_train = prepare_data_UEA(rep_ds_train, meta_csv, mapping_c_l, mode)
    end_train = time.time()
    #print('time cost for getting training data  : ' + str(end_train - start))
    X_test, X_s_test, masking_test, Y_test = prepare_data_UEA(rep_ds_test, meta_csv, mapping_c_l, mode)
    end_test = time.time()
    #print('time cost for getting testing data  : ' + str(end_test - end_train))
    X_sup, X_s_sup, masking_sup, Y_sup, X_unsup, X_s_unsup, masking_unsup, Y_unsup, n_classes = split_dataset(X_train,
                                                                                                          X_s_train,
                                                                                                    masking_train,
                                                                                                          Y_train,
                                                                                                          sup_ratio,
                                                                                                            split_strategy)  # split the training set into labeled and unlabed samples
    end_split = time.time()
    #print('time cost for getting splitting data  : ' + str(end_split - end_test))
    dataset = {}
    dataset.update({'X_train': X_train})
    dataset.update({'X_test': X_test})
    dataset.update({'X_s_train': X_s_train})
    dataset.update({'X_s_test': X_s_test})
    dataset.update({'masking_train': masking_train})
    dataset.update({'masking_test': masking_test})
    dataset.update({'Y_train': Y_train})
    dataset.update({'Y_test': Y_test})
    dataset.update({'X_sup': X_sup})
    dataset.update({'X_s_sup': X_s_sup})
    dataset.update({'masking_sup': masking_sup})
    dataset.update({'Y_sup': Y_sup})
    dataset.update({'X_unsup': X_unsup})
    dataset.update({'X_s_unsup': X_s_unsup})
    dataset.update({'masking_unsup': masking_unsup})
    dataset.update({'Y_unsup': Y_unsup})
    dataset.update({'n_classes': n_classes})

    return dataset
