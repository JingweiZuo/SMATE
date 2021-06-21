from utils.UEA_utils import *

from SMATE_model import SMATE
import time, os, math
import numpy as np
import pandas as pd
import random as rd
import argparse

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.core.protobuf import rewriter_config_pb2

K.clear_session()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.arithmetic_optimization = off
sess = tf.Session(config=config)
K.set_session(sess)


# add parameters

def get_parameters():
        parser = argparse.ArgumentParser(description="SMATE for Semi-Supervised MTS Learning")
        parser.add_argument('--enable_cuda', type=bool, default='True',
                            help='enable CUDA, default as True')
        parser.add_argument('--sup_ratio', type=float, default=1.0,
                            help='the supervised ratio, default as 1.0')
        parser.add_argument('--d_prime_ratio', type=float, default=0.5,
                            help='the ratio that calculates the hidden dimension size p_prime compared to the variable number M')
        parser.add_argument('--p_ratio', type=float, default=0.1,
                            help='the ratio that calculate the pool/sampling size P')
        parser.add_argument('--kernels', type=str, default="8,5,3",
                            help='the kernel/window size for the 3 convolutional block in spatial channel')
        parser.add_argument('--epochs', type=int, default=3000,
                            help='epochs, default as 3000')
        parser.add_argument('--rep_main', type=str, default='./Datasets/MTS-UEA/',
                            help='the root path of datasets')
        parser.add_argument('--ds_name', type=str, default='Cricket',
                            help='the UEA MTS dataset name')
        parser.add_argument('--outModelFile', type=str, default='./out/best_model',
                            help='save the model weights')
        
        args = parser.parse_args()
        print('Training configs: {}'.format(args))
        return args

def prepare_data(rep_main, ds_name, sup_ratio, pool_ratio, d_prime_ratio):
        rep_ds_train = rep_main + ds_name + "/output_train/"
        rep_ds_test = rep_main + ds_name + "/output_test/"
        meta_csv = "meta_data.csv"  # the meta data of training/testing set
        rep_output = rep_ds_train + "out/"  # output results, e.g., training loss, models
        os.system("mkdir -p " + rep_output)

        dataset = get_UEA_dataset(rep_ds_train, rep_ds_test, meta_csv, sup_ratio, mode='load')

        x_train = dataset['X_train']
        x_test = dataset['X_test']
        x_sup = dataset['X_sup']  # 3-D Array: N * T * M
        x_unsup = dataset['X_unsup']

        # Bacis Dataset Information and Model Configurations
        train_size = x_train.shape[0]
        T = x_train.shape[1]
        M = x_train.shape[2]
        pool_size = int(pool_ratio * T)
        d_prime = int(d_prime_ratio * M)
        n_class = dataset['n_classes']
        label_size = x_sup.shape[0]
        unlabel_size = x_unsup.shape[0]
        print("n_train is", train_size, "; n_test =", x_test.shape[0], "; T =", T, "; M =", M,
              "; n_class =",n_class)
        return dataset, T, M, pool_size, d_prime, n_class, label_size, unlabel_size

if __name__ == "__main__":
        args = get_parameters()

        n_epochs = args.epochs
        kernels = args.kernels

        rep_main = args.rep_main
        ds_name = args.ds_name
        sup_ratio = args.sup_ratio

        # Two hyper-parameters vary with datasets
        pool_ratio = args.p_ratio
        d_prime_ratio = args.d_prime_ratio

        dataset, T, M, pool_size, d_prime, n_class, label_size, unlabel_size = prepare_data(rep_main, ds_name, sup_ratio, pool_ratio, d_prime_ratio)

        x_train = dataset['X_train']
        y_train = dataset['Y_train']
        x_test = dataset['X_test']
        y_test = dataset['Y_test']
        x_sup = dataset['X_sup']  # 3-D Array: N * T * M
        x_unsup = dataset['X_unsup']
        y_sup = dataset['Y_sup']  # 1-D Array
        y_unsup = dataset['Y_unsup']

        '''=================================================== Model Training ========================================================'''
        # Build SMATE model
        smate = SMATE(T, M, n_class, label_size, unlabel_size,
                      y_sup, sup_ratio, pool_size, d_prime, args.outModelFile)

        smate.build_model()
        # smate.model.summary()
        # Train SMATE model
        t1 = time.time()
        smate.fit(n_epochs, x_train, x_sup, x_unsup)
        print("training time is {}".format(time.time() - t1))
        smate.model.load_weights(args.outModelFile)
        '''=================================================== Model Testing ========================================================'''
        # Test SMATE model on both supervised and semi-supervised classification
        smate.predict(x_train, y_train, x_test, y_test)
        # smate.predict_ssl(x_sup, y_sup, x_unsup, y_unsup, x_test, y_test) #semi-supervised prediction
