# CA-SFCN, a.k.a., nmsu-ijcai2020
Check the original Github repo [here](https://github.com/huipingcao/nmsu_yhao_ijcai2020) for the paper "A new attention mechanism to classify multivariate time series", IJCAI'20, by Yifan Hao and Huiping Cao. 

### UEA data processing

- We provide a [script](../Datasets/MTS-UEA/Preprocessing_MTS_UEA.ipynb) for transforming UEA data into CA-SFCN format

### Usage

The DATA_NAME is the name of preprocessed UEA dataset, which should be put into the folder "*./data*"

```
python fcn_ca_main.py <DATA_NAME> <ATTENTION_TYPE>
```

```
<ATTENTION_TYPE>: the method we can test. 
4 possible values: -1, 0, 1, 2.
-1 means SFCN (Stablized Fully-Convolutional Network without any attention)
0 means CA-SFCN (Cross-Attention Stablized Fully-Convolutional Network)
1 means GA-SFCN (Global-Attention Stablized Fully-Convolutional Network)
2 means RA-SFCN (Recurrent-Attention Stablized Fully-Convolutional Network)

For example, the command "python fcn_ca_main.py ges 0"
runs CA-SFCN method on ges Dataset.
```

