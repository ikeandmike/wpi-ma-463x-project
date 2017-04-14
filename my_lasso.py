import numpy as np
from sklearn import linear_model

def my_lasso(X, y, alpha=0.5):
    """Performs lasso on the data and returns the transformed data

    Args:
        X - the data to perform lassp upon (N,D)
        y - the class labels (N,)
        alpha - lasso hyperparam

    Returns: 
        modified data (N,?)
    """
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X,y)
    coef = clf.coef_   
    keep = coef > 0.0
    X_new = X[:,keep]
    return X_new

def test_lasso(X, y, alpha=0.5):
    """Performs lasso on the data and returns the transformed data

        This is for testing lasso on the data, it additionally returns 
        list of coeffiecents for charting/exploration.

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
    from data_utils import get_training
    X, y = get_training()
    X_new, coef = test_lasso(X,y)
    print 'pre lasso shape'
    print np.shape(X)
    print 'post lasso shape'
    print np.shape(X_new)

    import matplotlib.pyplot as plt
    plt.plot(coef)
    plt.ylabel('relative importance of predictors')
    plt.title('performing lasso on raw data')
    plt.show()

    from my_pca import my_pca
    X_pca = my_pca(X, 30)
    print 'post pca shape - and keeping first 5 principal dimensions'
    print np.shape(X_pca)
    X_pca_lasso, pca_coef = test_lasso(X_pca, y)
    print 'post lasso on pca shape'
    print np.shape(X_pca_lasso)

    plt.plot(coef)
    plt.ylabel('relative importance of predictors')
    plt.title('performing lasso post pca')
    plt.show()
