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

UEA_MTS_List = [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "CharacterTrajectories",
        "Cricket", #5
        #"DuckDuckGeese",
        #"EigenWorms",
        "Epilepsy",
        "EthanolConcentration",
        "ERing",#10
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        #"Handwriting",
        "Heartbeat",#15
        #"InsectWingbeat",
        #"JapaneseVowels",
        #"Libras",
        "LSST",
        "MotorImagery",#20
        "NATOPS",
        "PenDigits",
        "PEMS-SF",
        "PhonemeSpectra",
        #"RacketSports",#25
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",        
        #"UWaveGestureLibrary"
]

hyper_paras_map = {
        "ArticularyWordRecognition" : [10, 4], 
        "AtrialFibrillation" : [64, 2],
        "BasicMotions" : [10, 2],
        "CharacterTrajectories" : [30, 2],
        "Cricket" : [100, 4], #5
        #"DuckDuckGeese",
        #"EigenWorms",
        "Epilepsy" : [30, 2],
        "EthanolConcentration" : [50, 2],  
        "ERing": [5, 4],#10
        "FaceDetection" : [8, 16],  
        "FingerMovements" : [5, 8],  
        "HandMovementDirection" : [40, 4],  
        #"Handwriting",
        "Heartbeat" : [5, 16],  #15
        #"InsectWingbeat",
        #"JapaneseVowels",
        #"Libras",
        "LSST" : [4, 4],  
        "MotorImagery" : [100, 8],  #20
        "NATOPS" : [3, 6],
        "PenDigits" : [2, 2],
        "PEMS-SF" : [10, 64],  
        "PhonemeSpectra" : [30, 4],  
        #"RacketSports",#25
        "SelfRegulationSCP1" : [100, 6],  
        "SelfRegulationSCP2" : [100, 6],  
        "SpokenArabicDigits" : [10, 4],  
        "StandWalkJump" : [100, 4],        
        #"UWaveGestureLibrary"
}


sup_ratio = 1 # the supervised ratio in training set, 1 by defaut (fully supervised)
n_epochs = 500


'''=================================================== Prepare data ========================================================'''
## Prepare UEA data
rep_main = "../Datasets/MTS-UEA/"
ds_name = "SelfRegulationSCP1"
# Two hyper-parameters vary with datasets
pool_step = hyper_paras_map[ds_name][0]
d_prime = hyper_paras_map[ds_name][1]
rep_ds_train = rep_main + ds_name + "/output_train/"
rep_ds_test = rep_main + ds_name + "/output_test/"
meta_csv = "meta_data.csv"  # the meta data of training/testing set
rep_output = rep_ds_train + "out_results/"  # output results, e.g., training loss, models
os.system("mkdir -p " + rep_output)

dataset = get_UEA_dataset(rep_ds_train, rep_ds_test, meta_csv, sup_ratio, mode='load')

x_train = dataset['X_train']
y_train = dataset['Y_train']
x_test = dataset['X_test']
y_test = dataset['Y_test']
x_sup = dataset['X_sup']  # 3-D Array: N * L * D
x_unsup = dataset['X_unsup']
y_sup = dataset['Y_sup']  # 1-D Array
y_unsup = dataset['Y_unsup']


# Bacis Dataset Information and Model Configurations
train_size = x_train.shape[0] 
L = x_train.shape[1]
data_dim = x_train.shape[2]
n_class = dataset['n_classes']
label_size = x_sup.shape[0]
unlabel_size = x_unsup.shape[0]

print("n_train is", train_size, "; n_test =", x_test.shape[0], "; L =", L, "; D =", data_dim, 
      "; n_class =",n_class)

# Build SMATE model
smate = SMATE(L, data_dim, n_class, label_size, unlabel_size, 
              y_sup, sup_ratio, pool_step, d_prime)

smate.build_model("step_1")
#smate.model.summary()
# Train SMATE model
t1 = time.time()
smate.fit(n_epochs, x_train, x_sup, x_unsup)
print("training time is {}".format(time.time() - t1))

# Test SMATE model on both supervised and semi-supervised classification
smate.predict(x_train, y_train, x_test, y_test)
#smate.predict_ssl(x_sup, y_sup, x_unsup, y_unsup, x_test, y_test) #semi-supervised prediction



