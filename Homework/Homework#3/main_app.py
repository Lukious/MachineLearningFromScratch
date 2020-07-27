# from anaconda 3
import numpy as np
import pandas as pd
# ...

import NBS

# data logging
data = np.array(pd.read_csv('data.csv', sep='\t', lineterminator='\r', header=None))
label_str = np.array(pd.read_csv('label.csv', header=None))
label = np.array(NBS.ch_label(label_str))

# data shuffle
idx_shuffle = np.random.permutation(len(label))
label = label[idx_shuffle]
data = data[idx_shuffle,:]


# feature normalization 
nomal_data = NBS.feature_normalization(data)

# spilt data for testing
spilt_factor = 100
train_data,test_data,train_label,test_label = NBS.spilt_data(data,label,spilt_factor)

# get train parameter of nomal distribution 
mu_train, sigma_train = NBS.get_normal_parameter(train_data,train_label,3)

# get nomal distribution probability of each feature based on train feature 
prob,pi = NBS.prob(mu_train,sigma_train,test_data,test_label)

# classification using prob
estimation = NBS.classifier(prob)

# get accuracy
acc, acc_s = NBS.acc(estimation,test_label)

# print result
print('accuracy is ' + str(acc) + '% ! ! ')
print('the number of correct data is ' + str(acc_s) + ' of ' + str(len(test_label)) + ' ! ! ')

