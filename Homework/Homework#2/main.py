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
        return sampled_cen #return init centers

    def Assignment(self, ): # code for overall process
        return clusters , iteration #return clustrrs and iterations

    def Update(self,centroids, prev_centro, iters): # Update as a teration checker and centroid assigmenrtor
    
    def Train(self, ):  # Train for get result and Processing overall kmeans workings
        return result # return result

    def get_UD(self,data, centroids, clusters): # Get Uclideint distance
        return clusters     # return whole clustsers k sub-clusters

if __name__ == '__main__': # Start from main
    colorlist = ['r','c','k','g','m','b','y'] # Set color list (set this pallet because white and yellow is hard to congize)
    data = pd.read_csv('data.csv') # load data
    model1 = kmeans_(k=3, data=data, iteration=iteration) # implemented model init setting
    plt.scatter(result_x,result_y,c=str((colorlist[i]))) #plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot

    model2 = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data,predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'],data['Sepal length'],c=data['predict'],alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot