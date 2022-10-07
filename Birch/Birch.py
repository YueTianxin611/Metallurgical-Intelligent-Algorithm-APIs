from sklearn.cluster import Birch


def brc_cal(dx,branching_factor,n_clusters,threshold,compute_labels,copy):
    X = dx.values
    brc = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold,
                compute_labels=compute_labels,copy=copy)
    brc.fit(X)
    center = brc.subcluster_centers_
    lable = brc.subcluster_labels_
    lables = brc.labels_
    return center,lable,lables

