from SMATE_model import *

import time, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K


tf.get_logger().setLevel('ERROR')
K.clear_session()

sup_ratio = 1 # the supervised ratio in training set, 1 by defaut (fully supervised)
n_epochs = 30

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
        "BasicMotions" : [100, 2],
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

'''=================================================== Prepare data ========================================================'''
## Prepare UEA data
rep_main = "../Datasets/MTS-UEA/"
ds_name = "EthanolConcentration"
rep_ds_train = rep_main + ds_name + "/output_train/"
rep_ds_test = rep_main + ds_name + "/output_test/"
meta_csv = "meta_data.csv"  # the meta data of training/testing set
rep_output = rep_ds_train + "out_results/"  # output results, e.g., training loss, models
os.system("mkdir -p " + rep_output)

dataset = get_UEA_dataset(rep_ds_train, rep_ds_test, meta_csv, sup_ratio, mode='load')

def running_time(dataset, sample_rate, train_rate):
    x_train = dataset['X_train']
    x_sup = dataset['X_sup']  # 3-D Array: N * L * D
    x_unsup = dataset['X_unsup']
    y_train = dataset['Y_train']
    y_sup = dataset['Y_sup']
    
    nbr_sample = int(sample_rate * x_train.shape[1])
    nbr_ts_instance = int(train_rate * x_train.shape[0])
    
    x_train = resample_dataset(x_train, nbr_sample)[: nbr_ts_instance]
    y_train = y_train[: nbr_ts_instance]

    # Bacis Dataset Information and Model Configurations
    train_size = x_train.shape[0] 
    L = x_train.shape[1]
    data_dim = x_train.shape[2]
    n_classes = dataset['n_classes']

    label_size = x_sup.shape[0]
    unlabel_size = x_unsup.shape[0]
    
    
    print("n_train is", train_size, "; L =", L, "; D =", data_dim, 
          "; n_class =",n_classes)

    # Two hyper-parameters vary with datasets
    pool_step = hyper_paras_map[ds_name][0]
    d_prime = hyper_paras_map[ds_name][1]
    
    # Build SMATE model
    
    smate = SMATE(L, data_dim, n_classes, label_size, unlabel_size, 
                  y_sup, sup_ratio, pool_step, d_prime)

    smate.build_model()

    # Train SMATE model
    start = time.time()
    smate.fit(n_epochs, x_train, x_sup, x_unsup)
    print("Training Time for sample_rate (%f2) train_rate (%f2)  is %d" 
          %(sample_rate, train_rate, time.time() - start))
    #K.clear_session()
    return time.time() - start
    
def resample_dataset(x, nbr_sample):
    x_sampled = np.zeros(shape=(x.shape[0], nbr_sample, x.shape[2])) # N' * L * D 
    from scipy import signal
    for i in range(x.shape[0]):
        f = signal.resample(x[i], nbr_sample, axis = 0)
        x_sampled[i] = f
    return x_sampled

def save_running_time(ds_name, dataset, save_path, sample_rate, train_rate):
    df_time = pd.DataFrame(data = np.zeros((1, 4)), columns = ['Dataset', "train_rate", 'sample_rate', 'run_time'])
    run_time = running_time(dataset, sample_rate, train_rate)
    df_time['Dataset'] = ds_name
    df_time['train_rate'] = train_rate
    df_time['sample_rate'] = sample_rate
    df_time['run_time'] = run_time
    if not os.path.exists(save_path + "running_time.csv"):
        df_time.to_csv(save_path + "running_time.csv", index=False)
    else:
        res = pd.read_csv(save_path + "running_time.csv")
        res = pd.concat((res, df_time))
        res.to_csv(save_path + "running_time.csv", index=False)

if __name__ == '__main__':
    # output training time for different sample_rate & train_rate
    # A) train_rate = 1
    train_rate = 1
    for sample_rate in np.linspace(0.3, 1, 8):
        save_running_time(ds_name, dataset, rep_output, sample_rate, train_rate)
