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
parser.add_argument("--filename", type=str, default="Trace")
parser.add_argument("--horizon", type=float, default=0.2)
parser.add_argument("--stride", type=float, default=0.1)
parser.add_argument("--seed1", type=int, default=4)
parser.add_argument("--seed2", type=int, default=3)
parser.add_argument("--layer_id", type=int, default=6)#max is 6

args = parser.parse_args()
filename=args.filename
horizon=args.horizon
stride=args.stride
seed1=args.seed1
seed2=args.seed2
layer_id=args.layer_id

sys.stdout=open("clf_tfl_f2_"+str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+"_"+str(layer_id)+".log","w")

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
        if int(x.size()[2]%2)==0:
            x = self.maxpool(x)
        else: 
            x = self.maxpool_padding(x)
        return x    
        
    def forward(self, x_forecast):
        x = F.relu(self.conv1(x_forecast))
#        print("conv1: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max1: ", x.size())
        
        x = F.relu(self.conv2(x))
#        print("conv2: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max2: ", x.size())

        x = F.relu(self.conv3(x))
#        print("conv3: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max3: ", x.size())

        x = F.relu(self.conv4(x))
#        print("conv4: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max4: ", x.size())

        x = F.relu(self.conv5(x))
#        print("conv5: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max5: ", x.size())

        x = F.relu(self.conv6(x))
#        print("conv6: ", x.size())

        x = self.apply_maxpooling(x)
#        print("max6: ", x.size())

        x=x.view(x.size()[0],-1)
#        print("x.size() just after flattening: ", x.size())
        
        self.fc1 = nn.Linear(x.size()[1], 200)
        x = F.relu(self.fc1(x))
        
#        print("final state before the out: ", x.size())
        x=F.relu(self.fc2(x))
        
        out=self.output(x)
        return out

#we just care for the conv layers below, so layer ids to the nonLinearNet could be as follows :1 is first conv, :3 is the first three conv layers
#also we need to do maxpooling as the model would just get the output from the conv layers but not apply the maxpooling by itself.

#list(tf_model.children())
#Out[37]: 
#[Conv1d(1, 8, kernel_size=(3,), stride=(1,), padding=(1,)),
# Conv1d(8, 16, kernel_size=(3,), stride=(1,), padding=(1,)),
# Conv1d(16, 32, kernel_size=(3,), stride=(1,), padding=(1,)),
# Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,)),
# Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,)),
# Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,)),
# MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
# MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
# Linear(in_features=256, out_features=200, bias=True),
# Linear(in_features=512, out_features=200, bias=True),
# Linear(in_features=1024, out_features=200, bias=True),
# Linear(in_features=200, out_features=100, bias=True),
# Linear(in_features=100, out_features=4, bias=True),
# Linear(in_features=1280, out_features=200, bias=True)]

class nonLinearNet(torch.nn.Module):
    def __init__(self, layer_id):
        super(nonLinearNet, self).__init__()           
        self.maxpool_padding = nn.MaxPool1d(2,padding=1)
        self.maxpool = nn.MaxPool1d(2,padding=0)
        
        self.fc2=nn.Linear(200,100)
        self.output = nn.Linear(100,nb_classes)

    def apply_maxpooling(self, x):
        if int(x.size()[2]%2)==0:
            x = self.maxpool(x)
        else: 
            x = self.maxpool_padding(x)
        return x    
        
    def forward(self, x):
        layers=list(tf_model.children())
        for i in range(layer_id):
            x = F.relu(layers[i](x))
            x = self.apply_maxpooling(x)
#        print("x.size after loop: ",x.size())
        
        x=x.view(x.size()[0],-1)
        self.fc1 = nn.Linear(x.size()[1], 200)
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        out=self.output(x)
        return out
    
def optimize_network(x_batch_class, y_batch_class):
    y_hat_classification = nnet(x_batch_class.float())
    loss_classification = criterion_classification(y_hat_classification, y_batch_class.long())
    optimizer.zero_grad()
    loss_classification.backward()
    optimizer.step()
    return loss_classification.item()    

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
max_acc_possible = 1-(sum([list(y_test).count(x) for x in list(set(np.unique(y_test))-set(np.unique(y_train)))])/len(y_test))

x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
        
tf_model = fNet(int(horizon*x_train.shape[-1])).to(device)#doesn't make much sense here, but legacy code! so just passing the horizon for the initialization to go through
#model.load_state_dict(torch.load("/home/shayan/tfl_2/4_3_Trace_0.2_0.1.pt",map_location='cpu'))
tf_model.load_state_dict(torch.load("/home/shayan/tfl_2/forecasting_results_small_step_size/"+str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+".pt",map_location='cpu'))#,map_location=lambda storage, loc: storage.cuda()))
#                                    ,map_location=lambda storage, loc: storage.cuda()))
#tf_model.load_state_dict(torch.load(str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+".pt",map_location='cpu'))

for param in tf_model.parameters():
    param.requires_grad = False

nnet = nonLinearNet(layer_id).to(device)

print("here--->",next(tf_model.parameters()).is_cuda, next(nnet.parameters()).is_cuda)

criterion_classification = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nnet.parameters(), lr=1e-4)    

batch_size = 4
accuracies=[]

for t in range(5000):
    closses=[]
#    scheduler.step()
    indexes=np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train=x_train[indexes]
    y_train=y_train[indexes]
    
    x_train_batch=x_train[y_train!=-1]
    y_train_batch=y_train[y_train!=-1]
    
    for i in range(0,x_train_batch.shape[0],batch_size):
        if i+batch_size<=x_train_batch.shape[0]:
            closs = optimize_network(x_train_batch[i:i+batch_size], y_train_batch[i:i+batch_size])
            closses.append(closs)
        else:
            closs = optimize_network(x_train_batch[i:], y_train_batch[i:])
            closses.append(closs)
            
    val_acc = accuracy_score(np.argmax(nnet(x_test.float()).cpu().detach().numpy(),1),y_test.long().cpu().numpy())
    accuracies.append(val_acc)
    print("Epoch: ",t,"| Accuracy: ",val_acc, "/",max(accuracies),"/",max_acc_possible, "| Avg. c.e.loss: ", np.mean(closses), flush=True)
    if val_acc==1.0:
        break;
            
#model.output=nn.Identity(model.output.in_features,model.output.in_features)
