import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 직접적인 GMM 및 EM 라이브러리를 제외한 함수는 설치해서 자유롭게 import 가능

cluster_color = ['red','green','pink'] # 색깔은 자유롭게 설정

seed_num = 777
np.random.seed(seed_num) # seed setting

iteration = 100 # E-step과 M-step을 반복할 횟수

# 기본 변수만 제공, 자유롭게 구현 가능 (e.g. def 안에 def 작성 가능)


class EM_BASED_GMM_PARAM:
    def __init__(self, K, data, iteration):
        self.K = K
        self.data = data
        self.iteration = iteration

    def Initialization(self, ):  # 1. initialize mean, sigma, pi(initial probability)
        # your code here
        return something

    def Expectation(self, ):  # 2. Expectation step
        # your code here
        return something

    def Maximization(self, ): # 3. Maximization step
        # your code here
        return something

    def Train(self, ): # 4. Clustering, 10 point
        # your code here
        return something


    # 이외에, 자기가 원하는 util 함수 작성하여 사용가능

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    idx_shuffle = np.int32(pd.read_csv('shuffle_idx.csv',header=None)).reshape(data.shape[0]) # for data shuffle
    data = data[idx_shuffle,:]
    model = EM_BASED_GMM_PARAM(K=3, data=data, iteration=iteration)

    plt.scatter() # plot cluster final result
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title("Estimation")
    plt.show()