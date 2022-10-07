# -*- coding: utf-8 -*-
"""
Created on Tue nov 5 14:40:06 2019

@author: YTX

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

