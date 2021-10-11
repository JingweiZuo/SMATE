#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:24:31 2019

@author: shayan
"""
import os
import pandas as pd

#read the results from here: https://docs.google.com/spreadsheets/d/1nKD2mK-0vyY9Aa_s-uFbkpuF42mG-FZ8O7_iHkk4HIo/edit?usp=sharing
#drop the column of forecaster with steplength scheduler.
#and then find the best configuration that is the one that lead to the minimum mse on the forecasting problem, the conf is characterized with the stride and horizon
df=pd.read_csv("results_with_forecasters.csv")
df=df.drop(df.columns[-2], axis=1)#already sorted
confs_to_be_run={}
for i in range(0,len(df),6):
    print(i)
    min_so_far=df.values[i][-1]
    index_of_min_so_far=i
    print(df.values[i:i+6])
    for j in range(i,i+6):#to get the sparsity=3, horizon=2, 6 results per combination of seeds
        cur_min=df.values[j][-1]
        if cur_min<min_so_far:
            min_so_far=cur_min
            index_of_min_so_far=j
            
    filename = df.values[index_of_min_so_far][0]
    seed1 = df.values[index_of_min_so_far][1]
    seed2 = df.values[index_of_min_so_far][2]
    horizon = df.values[index_of_min_so_far][3] 
    stride = df.values[index_of_min_so_far][4]
    min_mse= df.values[index_of_min_so_far][5]
    
#    print(filename, seed1, seed2, horizon, stride)
    confs_to_be_run[filename, seed1, seed2, horizon, stride]=min_mse

tobebashed = """#!/usr/bin/env bash
#SBATCH --job-name=jobname                           
#SBATCH --output=jobname%j.log                       
#SBATCH --partition=CPU 
#SBATCH -x node-[042,105,043,106,039,041]
set -e
source /home/shayan/anaconda3/bin/activate /home/shayan/anaconda3/envs/strawberries
cd $PWD
srun /home/shayan/anaconda3/envs/strawberries/bin/python3 tfl_2.py"""


counter=0
for conf in confs_to_be_run.keys():
    filename, seed1, seed2, horizon, stride = conf[0], conf[1], conf[2], conf[3], conf[4]
    for layer_id in range(1,7):#conv1,...,conv6        
        tobebashed_final=tobebashed.replace("jobname","clf_tfl_"+str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+"_"+str(layer_id))
        args=" --filename "+str(filename)+" --horizon "+str(horizon)+" --stride "+str(stride)+" --seed1 "+str(seed1)+" --seed2 "+str(seed2)+" --layer_id "+str(layer_id)+"\n"
        print(args[:-2])
        tobebashed_final=tobebashed_final+args
        print(tobebashed_final)
        counter+=1
    
        with open("clf_tfl_"+str(seed1)+"_"+str(seed2)+"_"+str(filename)+"_"+str(horizon)+"_"+str(stride)+"_"+str(layer_id)+".sh","w") as fp:
            fp.write(tobebashed_final)

filenames=os.listdir()
filenames=[filename for filename in filenames if ".sh" in filename]
for filename in filenames:
    os.system("sbatch "+filename)
