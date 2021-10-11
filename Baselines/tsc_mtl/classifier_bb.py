#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:02:46 2019

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
parser.add_argument("--seed1", type=int, default=4)
parser.add_argument("--seed2", type=int, default=3)

args = parser.parse_args()
filename=args.filename
seed1=args.seed1
seed2=args.seed2

sys.stdout=open("clf_bb_"+str(seed1)+"_"+str(seed2)+"_"+filename+".log","w")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class cNet(torch.nn.Module): 
    def __init__(self):
        super(cNet, self).__init__()
        self.conv1 = nn.Conv1d(x_train.shape[1], 128, 9, padding=(9 // 2))
        self.bnorm1 = nn.BatchNorm1d(128)        
        self.conv2 = nn.Conv1d(128, 256, 5, padding=(5 // 2))
        self.bnorm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=(3 // 2))
        self.bnorm3 = nn.BatchNorm1d(128)        
        self.classification_head = nn.Linear(128, nb_classes)
        
    def forward(self, x_class):
#        print("x_class.size--->",x_class.size())
        b1 = F.relu(self.bnorm1(self.conv1(x_class)))
#        print("b1.size--->",b1.size())
        b2 = F.relu(self.bnorm2(self.conv2(b1)))
#        print("b2.size--->",b2.size())
        b3 = F.relu(self.bnorm3(self.conv3(b2)))
#        print("b3.size--->",b3.size())
        classification_features = torch.mean(b3, 2)#(64,128)#that is now we have global avg pooling, 1 feature from each conv channel
#        print("classification_features.size--->",classification_features.size())

        classification_out=self.classification_head(classification_features)
#        print("classification_out.size()--->",classification_out.size())
        return classification_out

    
def optimize_network(x_batch_class, y_batch_class):
    y_hat_classification = c_model(x_batch_class.float())
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

c_model = cNet().to(device)

criterion_classification = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(c_model.parameters(), lr=1e-4)    

batch_size = 4
accuracies=[]

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
best_model_wts = copy.deepcopy(c_model.state_dict())
best_val_acc = 0.0

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
            
    val_acc = accuracy_score(np.argmax(c_model(x_test.float()).cpu().detach().numpy(),1),y_test.long().cpu().numpy())
    accuracies.append(val_acc)
    print("Epoch: ",t,"| Accuracy: ",val_acc, "/",max(accuracies),"/",max_acc_possible, "| Avg. c.e.loss: ", np.mean(closses), flush=True)
    if val_acc==1.0:
        break;
