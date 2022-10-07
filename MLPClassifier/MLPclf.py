from sklearn.neural_network import MLPClassifier


def MLP_clf_cla(dx,dy,hidden_layer_sizes,activation,solver,alpha,batch_size,learning_rate,learning_rate_init,power_t,max_iter,
                shuffle,tol,verbose,warm_start, momentum, nesterovs_momentum,early_stopping,validation_fraction,
                beta_1,beta_2,epsilon,n_iter_no_change):
    X = dx.values
    Y = dy.values
    clf=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size,
                      learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle,
                      random_state=None, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                      early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

    #clf=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size,
                      #learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle,
                      #random_state=None, tol=tol, verbose=verbose, warm_start=warm_start,validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

    try:
        clf.fit(X,Y)
    except Exception as e:
        return str(e)
    prediction = clf.predict(X)
    score = clf.score(X,prediction)
    return score

