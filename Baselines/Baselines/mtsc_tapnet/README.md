# TapNet

This is a Pytorch implementation of Attentional Prototype Network for the task of (semi-supervised) classification of multivariate time series, as described in our AAAI 2020 paper:

**TapNet: Multivariate Time Series Classification with Attentional Prototype Network**

## Run the demo

```bash
python train.py
```

## Data

We use the latest multivariate time series classification dataset from [UAE archive](http://timeseriesclassification.com) with 30 datasets in wide range of applications.

The raw data is converted into npy data files in the following format:
* Training Samples (X_train.npy): an N by M by L tensor (N is the number of time series, M is the multivariate dimension, L is the length of time series),
* Train labels（y_train.npy）: an N by 1 vector (N is the number of time series)
* Testing Samples (X_test.npy): an N by M by L tensor (N is the number of time series, M is the multivariate dimension, L is the length of time series),
* Testing labels (y_test.npy): an N by 1 vector (N is the number of time series)

You can specify a dataset as follows:

```bash
python train.py --dataset NATOPS
```

(or by editing `train.py`)

The default data is located at './dataset'.
