####################################################################################
# File Name : bsw.py                                                               #  
# Date  : 2020/05/31                                                               #  
# OS : Windows 10                                                                  #  
# Author : Baek Su Whan                                                            #  
# Studnet ID : 2016722006                                                          #  
# -------------------------------------------------------------------------------  #  
# requirements : python 3.x / scipy / pandas /                                     #
#                                                                                  #
####################################################################################  
import numpy as np                              # import numpy for 2d matrix processing
import pandas as pd                             # pandas for overall processing
import matplotlib.pyplot as plt                 # for showing plots
from scipy.stats import multivariate_normal     # for get multivariate_normal

cluster_color = ['r','g','b']  # set scatter color

seed_num = 777                 # set random seed
np.random.seed(seed_num)       # seed setting
iteration = 100                # set number of iteration (e-step / m-step)

def Normalization(data): # Normalization function for get better resukt
    mean = np.mean(data, axis=0) # mean of each col
    std = np.std(data, axis=0) # sstandard deviation of each col
    result = [ ((data[:,i] - mean[i])/std[i] )  # Normalization formula
    for i in range(data.shape[1]) ] # Normalization
    return result # return result

class EM_BASED_GMM_PARAM:   ## GMM
    def __init__(self, K, data, iteration): #initalizing step
        self.K = K  # set numbor of k (this case 3)
        self.pdf = 0    # initialize probability density functino as 0
        self.data = data    #set data
        self.iteration = iteration #set iteration 
        self.mean, self.sig, self.pri = self.Initialization()  #set mu/sig/prior

    def Initialization(self,):  # first initialization step
        mean = self.data[np.random.randint(np.size(data,0), size=3)]  # get mean randomly
        sig = [np.eye(self.data.shape[1]) * np.random.rand() # set sig randomly
        for _ in range(self.K)]    # finish sigma update
        pri = np.ones([1, self.K]) / self.K  #for prior / prior setting
        return mean, sig, pri #return mean sigma prior 

    def Expectation(self, ):  # E step
        pdf = np.array([multivariate_normal.pdf(self.data,self.mean[i,:], self.sig[i]) # Set  probability density function
        for i in range(self.K)]).T # get numer of p(x)
        p_de = np.sum(pdf, 1) # get p(x) denominator values
        p_nor = pdf / p_de.reshape([self.data.shape[0], 1]) # calculate p(x)
        return p_nor # return return 

    def Maximization(self, ): 
        nk_mean,nk_sig,nk_pri = [] , [] , [] # init parameters

        for i in range(self.K): #for number of clusters
            mean_k = [sum(self.pdf[:,i]*self.data[:,j]) # set meak_k as a sum of mutipli pdf and data / sum of pdf
            for j in range(self.data.shape[1])] / sum(self.pdf[:,i]) # get mean k with as GMM formula
            nk_mean.append(mean_k)  # append the result to nk_mean
            sig_k = [sum(self.pdf[:,i] * (self.data-mean_k)[:,j] * (self.data-mean_k)[:,j]) / sum(self.pdf[:,j]) # update sigmal process
            for j in range(self.data.shape[1])] # update sigma value
            nk_sig.append(sig_k) # append the result of sigma to nk_sigma
            pri_k = sum(self.pdf[:,i]) / self.data.shape[0] # calculating prior
            nk_pri.append(pri_k) # append the result of prior to nk_pri
            re_nk_mean = np.array(nk_mean)  # data from change for training step
            re_nk_sig = np.array(nk_sig)**(1/2) # data from change for training step
            re_nk_pri = np.array(nk_pri) # data from change for training step
        return re_nk_mean, re_nk_sig, re_nk_pri #return values

    def Train(self, ): # Training step (clustering)
        for i in range(self.iteration): # for whole iteration
            self.pdf = self.Expectation()  # start estep
            self.mean, self.sig, self.pri = self.Maximization() #start mstep
        index = np.argmax(self.pdf,axis=1) # get result of traing
        return index #return pridict index
    
    def get_k(self, ): #get_k for get models k value
        return int(self.K) #return models number of clusters

if __name__ == '__main__': # start form here!
    data = pd.read_csv('./data.csv') # get data
    idx_shuffle = np.int32(pd.read_csv('./shuffle_idx.csv',header=None)).reshape(data.shape[0]) # get idx_shuffle
    data = np.array(data) # data from change
    data = data[idx_shuffle,:] # set idx_shuffle
    data = Normalization(data) # Normalization process 
    data = np.array(data).T # tranformation
    model = EM_BASED_GMM_PARAM(K=3, data=data, iteration=iteration) # Set Model
    index = model.Train() # Train and get cluster label

    for i in range(model.get_k()): # for whole resulted cluster
        temp_print = np.where(index == i)[0] # get traing result (each clusters)
        result_x = data[temp_print,0] # set x label
        result_y = data[temp_print,1] # set y label
        plt.scatter(result_x,result_y, c=cluster_color[i]) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)')  # set label
    plt.title("Estimation") # set title
    plt.show() #show plot