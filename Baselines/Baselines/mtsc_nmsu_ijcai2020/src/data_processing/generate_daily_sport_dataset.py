import numpy as np
np.random.seed(1000)

from scipy.io import loadmat

daily_sport = r""

''' Load train set '''
data_dict = loadmat(daily_sport + "daily_sport.mat")
X_train_mat = data_dict['X_train'][0]
y_train_mat = data_dict['Y_train'][0]
X_test_mat = data_dict['X_test'][0]
y_test_mat = data_dict['Y_test'][0]

y_train = y_train_mat.reshape(-1, 1)
y_test = y_test_mat.reshape(-1, 1)

var_list = []
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
max_nb_timesteps = var_list.max()
min_nb_timesteps = var_list.min()
median_nb_timesteps = np.median(var_list)

print('max nb timesteps train : ', max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)

X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    X_train[i, :, :var_count] = X_train_mat[i]

# ''' Load test set '''

X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_test_mat.shape[0]):
    var_count = X_test_mat[i].shape[-1]
    X_test[i, :, :var_count] = X_test_mat[i][:, :max_nb_timesteps]

# ''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

np.save(daily_sport + 'X_train.npy', X_train)
np.save(daily_sport + 'y_train.npy', y_train)
np.save(daily_sport + 'X_test.npy', X_test)
np.save(daily_sport + 'y_test.npy', y_test)
