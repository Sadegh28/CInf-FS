import os
import time
import pandas as pd
import numpy as np
import PyIFS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm


class CInfFS: 
    def __init__(self, num_clusters = 2, merge_coeff = 1): 
        self.num_clusters = num_clusters
        self.merge_coeff = merge_coeff

    def rank(self, X,y): 
        if self.num_clusters == 1: 
            selector = PyIFS.InfFS()
            [RANKED, _] = selector.infFS(X,y,alpha = 1, supervision= 1, verbose = True)
            return RANKED
        else: 
            kmeans = KMeans(n_clusters=self.num_clusters , random_state=0)
            kmeans.fit(np.transpose(X))
            a = kmeans.labels_
            l4=[]
            l5 = []
            for k in range(self.num_clusters):
                inf = PyIFS.InfFS()
                [RANKED, WEIGHT1] = inf.infFS(X[:,[i for i, e in enumerate(list(a)) if e == k]],y,alpha = 1, supervision= 1, verbose = True)
                original =[i for i, e in enumerate(list(a)) if e == k]
                WEIGHT1 *= len(original)/len(X)
                RANKED = np.argsort(WEIGHT1)
                RANKED = np.flip(RANKED,0)
                RANKED = RANKED.T
                
                
                MUTUAL = np.array([mutual_info_classif(X[:,index].reshape((-1,1)),y) for index,temp in enumerate(RANKED) ])
                
                scaler = MinMaxScaler()
                measure1 = scaler.fit_transform((np.real(WEIGHT1)).reshape((-1,1)))
                measure2 = scaler.fit_transform(MUTUAL)

                
                f = self.merge_coeff*(measure1) + (1-self.merge_coeff)*measure2 # formulate
                
                l4.append(np.sort(np.real(f)))
                l5.append([original[i] for i in RANKED])

            List1=l5[0]
            List2=l5[1]
            Weight1=l4[0]
            Weight1 = np.flip(Weight1,axis=0)
            Weight2=l4[1]
            Weight2 = np.flip(Weight2,axis=0)


            k = 0 
            i = 0
            j = 0

            Rank = []
            while k<len(Weight1)+len(Weight2)and i<len(Weight1) and j<len(Weight2):
                if Weight1[i]>Weight2[j] :
                    Rank.append(List1[i])
                    i+=1
                else:
                    Rank.append(List2[j])
                    j+=1
                k+=1
            while i<len(Weight1):
                Rank.append(List1[i])
                i+=1
            while j<len(Weight2):
                Rank.append(List2[j])
                j+=1

            return Rank











                        