

import numpy as np

def ch_label(label,label_str = ["'setosa'","'versicolor'","'virginica'"]):
    data_point = len(label)
    label_num = np.ones([data_point,1])
    for i in range(data_point):
        if(label[i]==label_str[0]):
            label_num[i] = 0
        elif(label[i]==label_str[1]):
            label_num[i] = 1
        else:
            label_num[i] = 2
    return label_num


def feature_normalization(data):
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    # you should get this parameter correctly
    nomal_feature = np.zeros([data_point,feature_num])
    ## your code here
            
    ## end
    return nomal_feature
        
def spilt_data(data,label,spilt_factor):
    feature_num = data.shape[1]
    data_point = data.shape[0]
    train_data = np.zeros([spilt_factor,feature_num])
    train_label = np.zeros([spilt_factor,1])
    test_data = np.zeros([data_point - spilt_factor,feature_num])
    test_label = np.zeros([data_point - spilt_factor,1]) 
    train_num = [i for i in range(spilt_factor)]
    test_num = [i for i in range(spilt_factor,len(label))]
    train_data = data[train_num,:]
    test_data = data[test_num,:]
    train_label = label[train_num]
    test_label = label[test_num]
    return train_data,test_data,train_label,test_label

## get_nomal_parameter
def get_normal_parameter(train_data,train_label,lable_num):
    ## parameter
    feature_num = train_data.shape[1]
    ## you should get this parameter correctly    
    mu = np.zeros([lable_num,feature_num])
    sigma = np.zeros([lable_num,feature_num])
    ## your code here

    ## end
    return mu,sigma

def prob(mu,sigma,data,label):
    ## parameter
    data_point = data.shape[0]
    lable_num = mu.shape[0]
    ## you should get this parameter correctly   
    prob = np.zeros([data_point,lable_num])
    pi = np.zeros([lable_num,1])
    ## your code here

    ## end
    return prob,pi

def classifier(prob):
    ## parameter
    data_point = prob.shape[0]
    ## you should get this parameter correctly 
    label = np.zeros([data_point])
    ## your code here

    ## end
    return label
        
def acc(est,gnd):
    total_num = len(gnd)
    acc = 0
    for i in range(total_num):
        if(est[i]==gnd[i]):
            acc = acc + 1
        else:
            acc = acc;
    return (acc / total_num)*100, acc
    
    

            
            
            
            
            
            
    
    
    
    
    
    
    
    