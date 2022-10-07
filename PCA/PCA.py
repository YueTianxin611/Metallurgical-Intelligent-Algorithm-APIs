from sklearn.decomposition import PCA


def pca_cal(df,n_components,copy,whiten,svd_solver,tol,iterated_power):

    X = df.values
    pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=None)
    try:
        pca.fit(X)
    except Exception as e:
        return str(e)
    return pca.components_,pca.explained_variance_,pca.explained_variance_ratio_,pca.singular_values_,pca.mean_,pca.n_components_,pca.noise_variance_

