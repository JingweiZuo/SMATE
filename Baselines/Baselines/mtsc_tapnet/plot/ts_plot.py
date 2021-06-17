import numpy as np
from matplotlib import pyplot as plt


# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
from utils import load_raw_ts

dataset = "LSST"
data_path = "../data/"
print("Loading dataset", dataset, "...")
features, labels, idx_train, idx_val, idx_test, nclass \
                                    = load_raw_ts(data_path, dataset=dataset, tensor_format=False)


fig, axes = plt.subplots(6, 1, figsize=(3, 5))

colors = ['g', 'r', 'c', 'm', 'y', 'k']
one_sample = features[0]
for i in range(len(one_sample)):
    ax = axes[i]
    dim = one_sample[i]
    ax.plot(dim, colors[i], linewidth=3)

plt.savefig("LSST.svg")
#plt.show()