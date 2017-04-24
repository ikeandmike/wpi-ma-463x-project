import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

def my_pca(X, d = 3):
    """Performs pca on the data and returns the transformed data

    Args:
        X - the data to perform pca upon (N,D)
        d - the number of dimnesions to project to (integer)

    Returns: 
        the pca components (D, D)
        explained variance (D)
    """
    N, D = np.shape(X)
    pca = PCA(n_components=d)
    X_scaled = preprocessing.scale(X)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced

if __name__ == '__main__':
    from data_utils import get_training
    X, y = get_training()
    N, D = np.shape(X)
    pca = PCA(n_components=D)
    X = preprocessing.scale(X)
    pca.fit(X)
    coef = pca.components_
    variances = pca.explained_variance_ratio_
    print 'shape of all the components (each component is a COLUMN)'
    print np.shape(coef)
    import matplotlib.pyplot as plt
    plt.plot(variances)
    plt.ylabel('percentage of variance explained')
    plt.show()
    ## an example of getting the top 3 components
    X_pca = my_pca(X, 3)
    print 'shape of data after projecting with top 3 components'
    print np.shape(X_pca)
