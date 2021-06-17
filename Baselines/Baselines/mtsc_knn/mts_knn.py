from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score

import numpy as np

# choose distance metric,
# Note: "dtw" in tslearn supports naturely MTS, however "euclidean distance" not support MTS in tslearn

#metric : {‘dtw’, ‘softdtw’, ‘euclidean’, ‘sqeuclidean’, ‘cityblock’, ‘sax’} (default: ‘dtw’)


''' 
    INPUT
    
    X : array-like, shape (n_ts, sz, d)
        Training data.
    y : array-like, shape (n_ts, )
        Target values.
'''


rep = "./datasets/multivariate/"
ds = "PenDigits"
ds_train = ds + '/' + ds + "_TRAIN3"
ds_test = ds + '/' + ds + "_TEST3"

def convert_mts(rep, dataset):
    seq = np.genfromtxt(rep + dataset, delimiter=' ', dtype=str, encoding="utf8")
    ids, counts = np.unique(seq[:,0], return_counts=True)

    No = ids.shape[0]
    D = seq.shape[1] - 3
    arr = np.asarray((ids, counts)).T
    Max_Seq_Len = np.max(arr[:,1].astype(np.int))

    out_X = np.zeros((No, Max_Seq_Len, D))
    out_Y = np.zeros((No, ))

    for idx, id in enumerate(ids):
        seq_cpy = seq[seq[:,0] == id]
        out_X[idx] = seq_cpy[:, 3:]
        out_Y[idx] = seq_cpy[0, 2]
    return out_X, out_Y


x_train, y_train = convert_mts(rep, ds_train)
x_test, y_test = convert_mts(rep, ds_test)

clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")

y_test_pred = clf.fit(x_train, y=y_train).predict(x_test)

print("the accuracy score of the testing data is : " + accuracy_score(y_test, y_test_pred))
