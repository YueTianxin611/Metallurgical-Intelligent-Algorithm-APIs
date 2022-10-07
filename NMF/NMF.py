from sklearn.decomposition import NMF


def nmf_cal(df,n_components, init, solver, beta_loss, tol, max_iter,
            alpha, l1_ratio, verbose, shuffle):
    X = df.values
    model = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol,
                max_iter=max_iter,alpha=alpha, l1_ratio=l1_ratio, verbose=verbose,
                shuffle=shuffle)
    try:
        model.fit_transform(X)
    except Exception as e:
        return str(e)
    return model.components_,model.reconstruction_err_,model.n_iter_

