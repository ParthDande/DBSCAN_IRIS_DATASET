# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:14:56 2023

@author: Parth
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris#to load the iris dataset


iris = load_iris()
df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
print(df.head())

pca = PCA(n_components=2)#we perform dimensionality reduction so that we can plot on graph
pca_df=pca.fit_transform(df)
print(pca_df.shape)

dbscan = DBSCAN(eps=0.4,min_samples=5)
cluster_labels =dbscan.fit_predict(pca_df)
x = pca_df[:,0]#to get x axis co-ordinates
y = pca_df[:,1]#to get y axis co-ordinates
plt.scatter(x,y, c =cluster_labels,cmap ='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')

#pca_df[:,0] give the x datapoints
#pca_df[:,1]give the y data points