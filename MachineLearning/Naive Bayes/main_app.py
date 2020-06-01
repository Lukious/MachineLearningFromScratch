# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:20:00 2017
☆☆☆ machine learning homework-1 (naive calssifier) ☆☆☆
"""
# from anaconda 3
import numpy as np
import pandas as pd

import naive as naive

# data logging
data = np.array(pd.read_csv('data.csv', sep='\t', lineterminator='\r', header=None))
label_str = np.array(pd.read_csv('label.csv', header=None))
label = np.array(naive.ch_label(label_str))

# data shuffle
idx_shuffle = np.random.permutation(len(label))
label = label[idx_shuffle]
data = data[idx_shuffle,:]


# feature normalization (10 point)
nomal_data = naive.feature_normalization(data)

# spilt data for testing
spilt_factor = 100
train_data,test_data,train_label,test_label = naive.spilt_data(data,label,spilt_factor)

# get train parameter of nomal distribution (20 point)
mu_train, sigma_train = naive.get_normal_parameter(train_data,train_label,3)

# get nomal distribution probability of each feature based on train feature (50 point)
prob,pi = naive.prob(mu_train,sigma_train,test_data,test_label)

# classification using prob (20 point)
estimation = naive.classifier(prob)

# get accuracy
acc, acc_s = naive.acc(estimation,test_label)

# print result
print('accuracy is ' + str(acc) + '% ! ! ')
print('the number of correct data is ' + str(acc_s) + ' of ' + str(len(test_label)) + ' ! ! ')

## total 100 point you can get 