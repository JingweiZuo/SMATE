from utils.basic_modules import *
from utils.UEA_utils import *

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as ll
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, add, GaussianNoise, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import to_categorical, plot_model, multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm, neighbors
from sklearn.semi_supervised import LabelSpreading


class SMATE:
    def __init__(self, L, data_dim, n_classes, label_size, unlabel_size, y_sup, sup_ratio, pool_step, d_prime, outModelFile):
        self.L = L
        self.data_dim = data_dim
        self.n_classes = n_classes
        self.label_size = label_size
        self.unlabel_size = unlabel_size
        self.train_size = label_size + unlabel_size
        self.y_sup = y_sup
        self.sup_ratio = sup_ratio
        self.pool_step = pool_step
        self.d_prime = d_prime
        self.outModelFile = outModelFile
        
    def build_model(self):
        y_sup_oneHot = to_categorical(self.y_sup, num_classes=self.n_classes) # n_sup * n_class (one-hot encoding)
        
        # linear mapping to low-dimensional space
        in_shape = (self.L, self.data_dim)  # the input shape of encoder
        self.model_e = encoder_smate(in_shape, self.pool_step, self.d_prime)
        #self.model_e = encoder_smate_rdp(in_shape, self.pool_step)
        #self.model_e = encoder_smate_se(in_shape, self.pool_step)
        
        h = self.model_e.output # batch_size * L

        # Init central class points

        idx_sup = np.array(range(self.label_size))
        h_sup = K.gather(h, idx_sup)
        proto_list = []
        for i in range(self.n_classes):
            idx = np.where(self.y_sup == i)[0]
            # compute the central point of each class
            class_repr = K.mean(K.gather(h_sup, idx), axis=0, keepdims=True)  # 1 * L
            proto_list.append(class_repr) # n_classes * L
        h_proto = ll.Concatenate(axis=0)(proto_list) # n_classes * L

        # Adjust central points

        dists_sup = euclidean_dist_mts(h_sup, h_proto) # n_sup * n_class
        dists_sum = K.sum(dists_sup, axis=1, keepdims=True) # normalize 'dists'
        dists_norm = dists_sup / dists_sum # n_sup * n_class (one-hot encoding)
        proba_sup = 1 - dists_norm
        proba_sup = multiply([y_sup_oneHot, proba_sup]) # # n_sup * n_class
        proba_sup = Lambda(lambda p: K.max(p, keepdims=True))(proba_sup) # n_sup * 1

        proto_list = []
        for i in range(self.n_classes):
            idx = np.where(self.y_sup == i)[0]
            class_repr = multiply([K.gather(h_sup, idx), K.gather(proba_sup,idx)]) #n_idx * L
            class_repr = K.sum(class_repr, axis=0, keepdims=True) # 1 * L
            proto_list.append(class_repr) # n_classes * L
        h_proto = ll.Concatenate(axis=0)(proto_list) # n_classes * L

        # Semi-supervised learning using unlabeled samples

        if self.sup_ratio != 1:
            idx_unsup = self.label_size + np.array(range(self.unlabel_size))
            h_unsup =K.gather(h, idx_unsup)

            dists_unsup = euclidean_dist_mts(h_unsup, h_proto) # n_unsup * n_class 
            dists_sum = K.sum(dists_unsup, axis=1, keepdims=True) # normalize 'dists'
            dists_norm = dists_unsup / dists_sum # n_sup * n_class (one-hot encoding)
            proba_unsup = 1 - dists_norm # get proba. distribution

            y_unsup_pseudo = K.argmax(dists_unsup, axis=1) # n_unsup * 1, get pseudo labels
            y_unsup_pseudo_oneHot = K.one_hot(y_unsup_pseudo, num_classes=self.n_classes) # n_unsup * n_class (one-hot encoding)

            proba_unsup = multiply([y_unsup_pseudo_oneHot, proba_unsup]) # # n_unsup * n_class, get probability over class
            proba_unsup = K.transpose(proba_unsup) # n_class * n_unsup

            proto_list = []
            for i in range(self.n_classes):
                proba_i = K.gather(proba_unsup, np.array([i])) # 1 * n_unsup 
                proba_i = K.transpose(proba_i) # n_unsup * 1
                class_repr = multiply([h_unsup, proba_i]) # n_usup * L
                class_repr = K.sum(class_repr, axis=0, keepdims=True) # 1 * L
                proto_list.append(class_repr) # n_classes * L
            h_proto_unsup = ll.Concatenate(axis=0)(proto_list) # n_classes * L

            # Adjust central points using unlabeled samples

            weight_sup = self.label_size / self.train_size
            weight_unsup = 1 - weight_sup
            h_proto = add([weight_sup * h_proto,  weight_unsup * h_proto_unsup])

        # Re-calculate the distance vector
        #dists_sup = euclidean_dist_mts(h_sup, h_proto) # n_sup * n_class
        
        # Define the auto-encoder models

        model_e_d = decoder_smate(self.model_e, self.L, self.data_dim, self.pool_step)

        # Reconstruction loss

        mts_in = self.model_e.input# batch_size * L * D
        mts_out = model_e_d.output

        rec_size = min(mts_in.shape[1], mts_out.shape[1])
        loss_rec = K.sqrt(K.sum(K.pow(mts_in[:, :rec_size, :] - mts_out[:, :rec_size, :], 2)) / self.train_size) # real value

        # Regularization loss

        dists_sum = K.sum(dists_sup, axis=1, keepdims=True) # normalize 'dists'
        dists_norm = dists_sup / dists_sum # n_sup * n_class (one-hot encoding)
        y_pred = 1 - dists_norm
        loss_reg = K.sum(categorical_crossentropy(y_pred, y_sup_oneHot)) / self.label_size

        loss_train = loss_rec + loss_reg
        model_e_d.add_loss(loss_train)
        opt = Adam(learning_rate=1e-05) # defaut LR: 1e-5
        model_e_d.compile(optimizer=opt)
        #plot_model(model_e_d, show_shapes=True)
        #model_e_d.summary()
        self.model = model_e_d
        
    def fit(self, n_epochs, x_train, x_sup, x_unsup):
        if self.sup_ratio == 1:
            x_fit = x_train
        else:
            x_fit = np.concatenate((x_sup, x_unsup), axis=0)

        validation_split = 0
        monitor_metric = 'loss'
        if x_fit.shape[0] > 100:
            validation_split = 0.2
            monitor_metric = 'val_loss'

        print('n_epochs=%d, batch_size=%d, n_sup=%d, n_sup=%d, steps=%d' % (
            n_epochs, self.train_size, self.label_size, self.unlabel_size, n_epochs))

        # checkpoint for best model
        checkpoint = ModelCheckpoint(self.outModelFile,
                                     monitor= monitor_metric,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        callbacks_list = [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(monitor= 'loss', patience=3, min_delta=0.0001, mode = 'auto')
        ]

        self.model.fit(
            x=x_fit,
            y=None,
            batch_size=self.train_size,
            epochs=n_epochs,
            verbose=0,
            callbacks=callbacks_list,
            validation_split=validation_split,
            validation_data=None,
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=True,
        )
        
        
    def predict(self, x_train, y_train, x_test, y_test):
        h_train = self.model_e.predict(x_train)
        h_test = self.model_e.predict(x_test)

        h_train = np.reshape(h_train, (h_train.shape[0], h_train.shape[1]*h_train.shape[2]))
        h_test = np.reshape(h_test, (h_test.shape[0], h_test.shape[1]*h_test.shape[2]))
        #SVM
        clf_svc = svm.SVC(kernel='linear')
        clf_svc.fit(h_train, y_train)
        acc_svm = accuracy_score(y_test, clf_svc.predict(h_test))

        clf_svc = svm.LinearSVC()
        clf_svc.fit(h_train, y_train)
        acc_svm_linear = accuracy_score(y_test, clf_svc.predict(h_test))
        print('acc_svm is ', max(acc_svm, acc_svm_linear))
        
    def predict_unsup(self, x_label, y_label, x_unlabel, x_test, y_test):
        h_label = self.model_e.predict(x_label)
        h_unlabel = self.model_e.predict(x_unlabel)
        h_test = self.model_e.predict(x_test)
        h_label = np.reshape(h_label, (h_label.shape[0], h_label.shape[1]*h_label.shape[2]))
        h_unlabel = np.reshape(h_unlabel, (h_unlabel.shape[0], h_unlabel.shape[1]*h_unlabel.shape[2]))
        h_test = np.reshape(h_test, (h_test.shape[0], h_test.shape[1]*h_test.shape[2]))
        # compute class centroids
        centroid_list = []
        for i in range(self.n_classes):
            idx = np.where(y_label == i)[0]
            # compute the central point of each class
            h_centroid = np.mean(h_label[idx], axis = 0, keepdims=True) # 1 * L
            centroid_list.append(h_centroid) 
        h_centroid = np.concatenate(centroid_list, axis=0) # n_classes * L

        y_unsups = []
        for h_i in h_unlabel:
            dist_array = np.sqrt(np.sum((h_i - h_centroid) ** 2, axis=0))
            pseudo_label = np.argmax(dist_array)
            y_unsups.append(pseudo_label)
        y_unsup = np.array(y_unsups)
        
        y_sup_unsup = np.concatenate([y_label, y_unsup])
        h_sup_unsup = np.concatenate([h_label, h_unlabel])
        #SVM
        clf_svc = svm.SVC(kernel='linear')
        clf_svc.fit(h_sup_unsup, y_sup_unsup)
        acc_svm = accuracy_score(y_test, clf_svc.predict(h_test))

        clf_svc = svm.LinearSVC()
        clf_svc.fit(h_sup_unsup, y_sup_unsup)
        acc_svm_linear = accuracy_score(y_test, clf_svc.predict(h_test))
        print('acc_svm is ', max(acc_svm, acc_svm_linear))
        
    def predict_ssl(self, x_sup, y_sup, x_unsup, y_unsup, x_test, y_test):
        
        ls_model = LabelSpreading(kernel='knn', n_neighbors=5)
        indices = np.arange(self.train_size)
        unlabeled_indices = indices[x_sup.shape[0]:]
        y_sup_unsup = np.concatenate([y_sup, y_unsup])
        y_sup_unsup_train = np.copy(y_sup_unsup)
        y_sup_unsup_train[unlabeled_indices] = -1
        
        x_fit = np.concatenate([x_sup, x_unsup], axis=0)
        h_fit = self.model_e.predict(x_fit)
        h_fit = np.reshape(h_fit, (h_fit.shape[0], h_fit.shape[1]*h_fit.shape[2]))
        ls_model.fit(h_fit, y_sup_unsup_train)
        y_unsup_pred = ls_model.transduction_[unlabeled_indices]

        #print("LabelSpread Accuracy is ", accuracy_score(y_unsup, y_unsup_pred))
        
        h_test = self.model_e.predict(x_test)
        h_test = np.reshape(h_test, (h_test.shape[0], h_test.shape[1]*h_test.shape[2]))
        
        #SVM
        clf_svc = svm.SVC(kernel='linear')
        y_fit_true = ls_model.transduction_
        clf_svc.fit(h_fit, y_fit_true)
        acc_svm = accuracy_score(y_test, clf_svc.predict(h_test))

        clf_svc = svm.LinearSVC()
        clf_svc.fit(h_fit, y_fit_true)
        acc_svm_linear = accuracy_score(y_test, clf_svc.predict(h_test))
        print('acc_svm is ', max(acc_svm, acc_svm_linear))

        
