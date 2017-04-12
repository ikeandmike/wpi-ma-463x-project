import numpy as np
from sklearn import linear_model

def my_lasso(X, y, alpha=1.0):
    """Performs pca on the data and returns the transformed data

    Args:
        X - the data to perform pca upon (N,D)
        y - the class labels (N,)
        alpha - lasso hyperparam

    Returns: 
        modified data (N,?)
        relative importance
    """
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X,y)
    coef = clf.coef_   
    keep = coef > 0.0
    X_new = X[:,keep]
    return X_new, coef

if __name__ == '__main__':
    from read_data import read_data
    from partition_data import partition_data
    X, y = read_data()
    X_tr, y_tr,_,_ = partition_data(X, y)
    X_new, coef = my_lasso(X_tr,y_tr)
    print 'pre lasso shape'
    print np.shape(X_tr)
    print 'post lasso shape'
    print np.shape(X_new)

    import matplotlib.pyplot as plt
    plt.plot(coef)
    plt.ylabel('relative importance of predictors')
    plt.title('performing lasso on raw data')
    plt.show()

    from my_pca import my_pca
    V, _ = my_pca(X_tr)
    X_pca = X_tr.dot(V[:,0:5]) # keeping top 5 for no good reason
    print 'post pca shape - and keeping first 5 principal dimensions'
    print np.shape(X_pca)
    X_pca_lasso, pca_coef = my_lasso(X_pca, y_tr)
    print 'post lasso on pca shape'
    print np.shape(X_pca_lasso)

    plt.plot(coef)
    plt.ylabel('relative importance of predictors')
    plt.title('performing lasso post pca')
    plt.show()

