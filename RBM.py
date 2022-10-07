import numpy as np
from sklearn.neural_network import BernoulliRBM

def rbm_cal(dx,dy,n_components,learning_rate,batch_size,n_iter,verbose):
    X = np.array(dx)
    Y = np.array(dy)
    rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate, batch_size=batch_size, n_iter=n_iter, verbose=verbose, random_state=None)
    try:
        rbm.fit(X)
    except Exception as e:
        return str(e)
    hidden = rbm.intercept_hidden_
    visible = rbm.intercept_visible_
    components = rbm.components_
    return hidden,visible,components



