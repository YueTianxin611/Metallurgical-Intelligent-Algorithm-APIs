# -*- coding: utf-8 -*-
"""
Created on Tue nov 5 14:40:06 2019

@author: ZHY

"""
import pandas as pd
from sklearn.cluster import AffinityPropagation

# 用高斯混合模型(GMM)的最大期望(EM)聚类


def affinitypropagation_cal(df,damping,max_iter,convergence_iter,copy, preference, affinity,verbose):
    X = df.values
    clu = AffinityPropagation(damping=damping,max_iter=max_iter,convergence_iter=convergence_iter,copy=copy,
                               preference=preference,affinity=affinity,verbose=verbose)
    clu.fit(X)
    cluster_centers_indices = clu.cluster_centers_indices_
    cluster_centers = clu.cluster_centers_
    labels = clu.labels_
    affinity_matrix = clu.affinity_matrix_
    n_iter = clu.n_iter_

    return cluster_centers_indices, cluster_centers, labels, affinity_matrix, n_iter
'''
if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\lq123\\Desktop\\F4_Stand_loaded_from_Observer=1.csv')
    cluster_centers_indices, cluster_centers, labels, affinity_matrix, n_iter = affinitypropagation_cal(df,damping=0.5,
        max_iter=200, convergence_iter=15, copy=True, preference=100, affinity='euclidean',verbose=False)
    print(cluster_centers_indices)
    print(cluster_centers)
    print(labels)
    print(affinity_matrix)
    print(n_iter)
'''
