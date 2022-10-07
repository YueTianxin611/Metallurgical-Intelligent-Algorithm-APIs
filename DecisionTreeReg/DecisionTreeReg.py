from sklearn.tree import DecisionTreeRegressor


def DecisionTree_reg_cal(dx,dy,criterion,splitter,max_depth,min_samples_split,min_samples_leaf,
                               min_weight_fraction_leaf,max_features,max_leaf_nodes,min_impurity_decrease,
                               min_impurity_split,presort):
    X = dx.values
    Y = dy.values
    clf=DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,
                               max_features=max_features,random_state=None,max_leaf_nodes=max_leaf_nodes,
                               min_impurity_decrease=min_impurity_decrease,
                               min_impurity_split=min_impurity_split,presort=presort)
    try:
        clf.fit(X,Y)
    except Exception as e:
        return str(e)
    prediction = clf.predict(X)
    score = clf.score(X,prediction)
    return score

