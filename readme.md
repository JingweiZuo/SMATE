## SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series
This is the code repo corresponding to the experiments conducted for the work "SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series".



## Requirements

- graphviz=2.40.1
- keras=2.2.4
- Matplotlib=3.2.1
- numpy=1.16.4
- pandas=0.24.2
- pydot=1.4.1
- scikit-learn=0.21.2
- tensorflow=1.14.0 with CUDA 10.2



## Files

#### Core

- **"./SMATE_model.py"**: the SMATE model building
- **"./utils/basic_modules.py"**: the components in SMATE architecture, as well as some baseline modules for comparing with the Spatial Modelling Block (SMB) 
- **"./utils/generic_utils.py"**: the generic operations on dataset, as well as basic calculations on MTS samples  (e.g., distance measures, normalizations ,etc.) 

#### Operation and Visualization

- **"./utils/UEA_utils.py"**: the processing on UEA datasets
- **"./utils/data_plot.py"**: the code allows visualizing the embedding space via TSNE plots 

#### Baseline Files

- **"./Baselines/"**: Except *WEASEL+MUSE* is implemented by Java for which the readers need to download MAVEN dependencies, we provide a Jupyter Notebook for testing on each other baselines including MLSTM-FCN, USRL, 1-NN based classifiers, TapNet. The enviroment configurations for each baselines are different, please check the details in the original papers.

### Dataset preprocessing

Due to the space constraint, we include only part of UEA-MTS datasets in this repo. However, you can find the full datasets on www.timeseriesclassification.com. The dataset provided on this site are in "arff" format, we provide the preprocessing code that you can find in ***[Preprocessing_MTS_UEA.ipynb](./Datasets/MTS-UEA/Preprocessing_MTS_UEA.ipynb)***

### Training & Testing

```python
python SMATE_classifier.py --d_prime_ratio 0.5 --p_ratio 0.1 --ds_name Cricket
```

