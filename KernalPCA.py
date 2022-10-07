from sklearn.decomposition import KernelPCA


def kernel_pca_cal(df,n_components, kernel, gamma, degree, coef0, kernel_params,
                alpha, fit_inverse_transform, eigen_solver, tol, max_iter,
                remove_zero_eig, copy_X, n_jobs):

    X = df.values
    transformer = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, kernel_params=kernel_params,
                alpha=alpha, fit_inverse_transform=fit_inverse_transform, eigen_solver=eigen_solver, tol=tol, max_iter=max_iter,
                remove_zero_eig=remove_zero_eig, random_state=None, copy_X=copy_X, n_jobs=n_jobs)
    try:
        transformer.fit(X)
    except Exception as e:
        return str(e)
    return transformer.lambdas_,transformer.alphas_,transformer.X_fit_
