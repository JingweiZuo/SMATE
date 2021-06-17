from tslearn.metrics import dtw
import numpy as np

def DTW_D(mts1, mts2):
    '''
        Dimension-dependent DTW measure
    :param mts1: array, shape=(L, D), L is MTS' length, D is MTS' dimension size
    :param mts2: array, shape=(L, D)
    :return: DTW-D score
    '''
    return dtw(mts1, mts2)

def DTW_I(mts1, mts2):
    '''
        Dimension-independent DTW measure
    :param mts1: array, shape=(L, D), L is MTS' length, D is MTS' dimension size
    :param mts2: array, shape=(L, D)
    :return: DTW-I score
    '''
    dtw_cum = 0
    for di in range(mts1.shape[1]):
        dtw_cum += dtw(mts1[:, di], mts2[:, di])
    return dtw_cum

def ED(mts1, mts2):
    '''

    :param mts1: array, shape=(L, D), L is MTS' length, D is MTS' dimension size
    :param mts2: array, shape=(L, D)
    :return: ED distance between MTS samples
    '''
    return np.sum((np.subtract(mts1, mts2))**2)

def DTW_D_D(mts1, mts2):
    '''

    :param mts1: array, shape=(L, D), L is MTS' length, D is MTS' dimension size
    :param mts2: array, shape=(L, D)
    :return: DTW/ED for dimension-dependent mts distance measure
    '''
    epsilon = 1**(-7)
    return DTW_D(mts1, mts2) / (ED(mts1, mts2) + epsilon)

def ssl_self_training():

    return 0
