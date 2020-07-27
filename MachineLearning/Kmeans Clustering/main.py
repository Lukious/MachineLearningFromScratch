####################################################################################
# File Name : bsw.py                                                               #  
# Date  : 2020/05/31                                                               #  
# OS : Windows 10                                                                  #  
# Author : Baek Su Whan                                                            #  
# Studnet ID : 2016722006                                                          #  
# -------------------------------------------------------------------------------  #  
# requirements : python 3.x / numpy / pandas / sklearn                             #
#                                                                                  #
####################################################################################   

import random                       # import random for random samping first centroids
import numpy as np                  # import numpy for 2d matrix processing
import pandas as pd                 # pandas for overall processing
import matplotlib.pyplot as plt     # for showing plots
import warnings                     # defrecation issue handler

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.")  # print general error  
    pass

LIMITION = 900  # Limit the maximum iteration(for get result much faster)
seed_num = 777  # set random seed
np.random.seed(seed_num) # seed setting
iteration = 300 # if value unchage untill 300 times

class kmeans_:
    def __init__(self, k, data, iteration): # initalize
        self.k = k # number of cluster
        self.data = data    # data
        self.iteration = iteration # set iteration [300]

    def Centroids(self, ):  # Set initali centorids
        data = self.data.to_numpy() # get data and change numpy for sampling
        idx = np.random.randint(int(np.size(data,0)), size=int(self.k)) # get random index by using randint
        sampled_cen = data[idx,:] # sampling...
        return sampled_cen #return init centers

    def Assignment(self, ): # code for overall process
        data = self.data.to_numpy() # change data to numpy for future processing
        cen = self.Centroids() # Get initial centroids
        prev_centro = [[] for i in range(self.k)] # get prev_cnetroid for comparing
        iters = 0 # set intertionflag 0
        warnings.simplefilter(action="ignore", category=FutureWarning) #ignore warnings check return part of Update
        while self.Update(cen, prev_centro, iters) is not True: # cheking Update (for LIMITION)
            iters =  iters + 1 # update iteration counter
            clusters = [[] for i in range(self.k)] # Set cluster 
            old_result = [[] for i in range(self.k)] # Set prev_cluster for comparing
            clusters = self.get_UD(data, cen, clusters) # Set cluster (Part of Assignment function)
            idx = 0 # Set index
            for result in clusters: # for whole clusters
                prev_centro[idx] = cen[idx] # update centroids
                cen[idx] = np.mean(result, axis=0).tolist() # Get center mean
                idx = idx+1 # update index counter
            if np.array_equal(old_result,result) is True:   # Comparing 
                iters = 0 # if iteration is not same update iters to 0 (start from ground again)
            iteration = self.iteration # get iteration
            old_result = result # update result
        return clusters , iteration #return clustrrs and iterations

    def Update(self,centroids, prev_centro, iters): # Update as a teration checker and centroid assigmenrtor
        if iters > LIMITION: # compare for LIMITION (early stopping)
            return True # for let loop Assignmnet functino
        warnings.simplefilter(action='ignore', category=FutureWarning) #ignore warnings check return part of Update
        return prev_centro == centroids # Allocation
        # numpy issue for '==' DeprecationWarning 
        # https://stackoverflow.com/questions/44574679/python-deprecationwarning-elementwise-comparison-failed-this-will-raise-an
        # Numpy Warning verstion ISSUE Unslovable (Anyway still working)
    
    def Train(self, ):  # Train for get result and Processing overall kmeans workings
        itertaion = 0 # set interation 0 (init)
        result,iteration = self.Assignment() # get result and iteration
        self.iteration = itertaion # update iteration
        return result # return result

    def get_UD(self,data, centroids, clusters): # Get Uclideint distance
        for ins in data:  #for whole data
            mu = min([(i[0], np.linalg.norm(ins-centroids[i[0]])) \
                                for i in enumerate(centroids)], key=lambda t:t[1])[0] # Get uclidient distnace formula
            try: # exception processing
                clusters[mu].append(ins) #for all clusters append instance (as a asssignment functino)
            except KeyError: # exceptino handling
                clusters[mu] = [ins] #update case
        for result in clusters: #for all sub-clusters
            if not result: #Nan case
                result.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist()) # Samplilng and apeend sub-clusters
        return clusters     # return whole clustsers k sub-clusters

if __name__ == '__main__': # Start from main
    colorlist = ['r','c','k','g','m','b','y'] # Set color list (set this pallet because white and yellow is hard to congize)
    data = pd.read_csv('data.csv') # load data
    model1 = kmeans_(k=7, data=data, iteration=iteration) # implemented model init setting
    clustsers = model1.Train() #set clusters
    result = [] #result list for set diff colors
    for i in range(int(model1.k)): # for k case
        result = np.array(clustsers[i]) # i control for reslut
        result_x = result[:,0] # Assign x
        result_y = result[:,1] # Assign y
        plt.scatter(result_x,result_y,c=str((colorlist[i]))) #plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot

    model2 = KMeans(n_clusters=7, init='random', random_state=seed_num, max_iter=iteration).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data,predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'],data['Sepal length'],c=data['predict'],alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot