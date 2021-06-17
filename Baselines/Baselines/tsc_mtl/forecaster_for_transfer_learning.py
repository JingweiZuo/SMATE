#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 03:15:57 2019

@author: shayan
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
import copy
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="FaceFour")
parser.add_argument("--horizon", type=float, default=0.2)
parser.add_argument("--stride", type=float, default=0.1)
parser.add_argument("--seed1", type=int, default=100)
parser.add_argument("--seed2", type=int, default=101)

args = parser.parse_args()
filename=args.filename
horizon=args.horizon
stride=args.stride
seed1=args.seed1
seed2=args.seed2

sys.stdout=open(str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+".log","w")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fNet(torch.nn.Module):
    def __init__(self, horizon):
        super(fNet, self).__init__()
        self.conv1 = nn.Conv1d(x_train.shape[1], 8, 3, padding=(3 // 2))
        self.conv2 = nn.Conv1d(8, 16, 3, padding=(3 // 2))
        self.conv3 = nn.Conv1d(16, 32, 3, padding=(3 // 2))
        self.conv4 = nn.Conv1d(32, 64, 3, padding=(3 // 2))
        self.conv5 = nn.Conv1d(64, 128, 3, padding=(3 // 2))
        self.conv6 = nn.Conv1d(128, 256, 3, padding=(3 // 2))
        
        self.maxpool_padding = nn.MaxPool1d(2,padding=1)
        self.maxpool = nn.MaxPool1d(2,padding=0)
        
        self.fc1_1=nn.Linear(256,200)
        self.fc1_2=nn.Linear(512,200)
        self.fc1_3=nn.Linear(1024,200)
        
        self.fc2=nn.Linear(200,100)
        self.output = nn.Linear(100,horizon)

    def apply_maxpooling(self, x):
#        print("i am calld atleast: ",x.size())
        if int(x.size()[2]%2)==0:
            x = self.maxpool(x)
        else: 
            x = self.maxpool_padding(x)
        return x    
        
    def forward(self, x_forecast):
#        print("x_forecast.shape: ",x_forecast.size())
        x = F.relu(self.conv1(x_forecast))
#        print("Size after 1 conv: ",x.size())
        x = self.apply_maxpooling(x)
#        print("Size after 1 maxpooling: ",x.size())
        x = F.relu(self.conv2(x))
        x = self.apply_maxpooling(x)
#        print("Size after 2 maxpooling: ",x.size())
        x = F.relu(self.conv3(x))
        x = self.apply_maxpooling(x)
#        print("Size after 3 maxpooling: ",x.size())
        x = F.relu(self.conv4(x))
        x = self.apply_maxpooling(x)
#        print("Size after 4 maxpooling: ",x.size())
        x = F.relu(self.conv5(x))
        x = self.apply_maxpooling(x)
#        print("Size after 5 maxpooling: ",x.size())
        x = F.relu(self.conv6(x))
        x = self.apply_maxpooling(x)
#        print("Size after 6 maxpooling: ",x.size())
        
        x=x.view(x.size()[0],-1)
#        print("Size after flattening: ",x.size())
        
        if x.size()[1]==256:
            x=F.relu(self.fc1_1(x))
        elif x.size()[1]==512:
            x=F.relu(self.fc1_2(x))
        elif x.size()[1]==1024:
            x=F.relu(self.fc1_3(x))
            
#        print("Size after 1 fc: ",x.size())
        x=F.relu(self.fc2(x))
#        print("Size after 2 fc: ",x.size())
        out=self.output(x)
#        print("out--->",out.size())
        return out
    
def optimize_network(x_sliding_window, y_sliding_window):
    y_hat_forecasting = model(x_sliding_window.float())
    loss_forecasting = criterion_forecasting(y_hat_forecasting, torch.squeeze(y_sliding_window).float())
    optimizer.zero_grad()
    loss_forecasting.backward()
    optimizer.step()
    return loss_forecasting.item()
    

train=pd.read_csv("/home/shayan/ts/UCRArchive_2018/"+filename+"/"+filename+"_TRAIN.tsv",sep="\t",header=None)
test=pd.read_csv("/home/shayan/ts/UCRArchive_2018/"+filename+"/"+filename+"_TEST.tsv",sep="\t",header=None)

     
df = pd.concat((train,test))
y_s=df.values[:,0]
nb_classes = len(np.unique(y_s))
y_s = (y_s - y_s.min())/(y_s.max()-y_s.min())*(nb_classes-1)
df[df.columns[0]]=y_s

train, test = train_test_split(df, test_size=0.2, random_state=seed1)
train_labeled, train_unlabeled = train_test_split(train, test_size=1-0.1, random_state=seed2)            
train_unlabeled[train_unlabeled.columns[0]]=-1#Explicitly set all the instance's labels to -1

train_1=pd.concat((train_labeled,train_unlabeled))
x_train=train_1.values[:,1:]
y_train=train_1.values[:,0]

x_test=test.values[:,1:]
y_test=test.values[:,0]

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
x_test = (x_test - x_train_mean)/(x_train_std)

x_train=x_train[:,np.newaxis,:] 
x_test=x_test[:,np.newaxis,:]

#x_train=x_train[:,:,np.newaxis] 
#x_test=x_test[:,:,np.newaxis]

x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
        
model = fNet(int(horizon*x_train.shape[-1])).to(device)#last dimension is the length of the series
criterion_forecasting = nn.MSELoss()
 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    

def return_sliding_windows(X):
    xf=[]
    yf=[]
    for i in range(0,X.shape[2],int(stride*X.shape[2])):
        horizon1=int(horizon*X.shape[2])
        if(i+horizon1+horizon1<=X.shape[2]):
#            print("X===>",i,i+horizon)
#            print("Y===>",i+horizon,i+horizon+horizon)
            xf.append(X[:,:,i:i+horizon1])
            yf.append(X[:,:,i+horizon1:i+horizon1+horizon1])
    
    xf=torch.cat(xf)
    yf=torch.cat(yf)
    
    return xf,yf#.reshape(xf.shape[0]*xf.shape[1],xf.shape[2],xf.shape[3]),yf.reshape(yf.shape[0]*yf.shape[1],yf.shape[2],yf.shape[3])
        
batch_size=32
accuracies=[]

x_sliding_window, y_sliding_window = return_sliding_windows(x_train)

xtrain, xtest = train_test_split(x_sliding_window.cpu().numpy(), test_size=0.2, random_state=453)
ytrain, ytest = train_test_split(y_sliding_window.cpu().numpy(), test_size=0.2, random_state=453)

xtrain = torch.from_numpy(xtrain).to(device)
ytrain = torch.from_numpy(ytrain).to(device)
xtest = torch.from_numpy(xtest).to(device)
ytest = torch.from_numpy(ytest).to(device)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
best_model_wts = copy.deepcopy(model.state_dict())
best_mse = 100000

for t in range(1000):
    print(filename)
    scheduler.step()
    indexes=np.array(list(range(xtrain.shape[0])))
    np.random.shuffle(indexes)
    xtrain=xtrain[indexes]
    ytrain=ytrain[indexes]
    
    for i in range(0,xtrain.shape[0],batch_size):
        if i+batch_size<=xtrain.shape[0]:
            optimize_network(xtrain[i:i+batch_size], ytrain[i:i+batch_size])
        else:
            optimize_network(xtrain[i:], ytrain[i:])
    val_mse=mean_squared_error(model(xtest.float()).detach().cpu().numpy(),np.squeeze(ytest.detach().cpu().numpy()))
    print(str(seed1)+"_"+filename+"_"+str(horizon)+"_"+str(stride),"Epoch: ",t,"| f_loss: ",val_mse,flush=True)
    
    if val_mse < best_mse:
        best_mse = val_mse
        best_model_wts = copy.deepcopy(model.state_dict())   
        torch.save(best_model_wts, str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+".pt")
#model.output=nn.Identity(model.output.in_features,model.output.in_features)
