import numpy as np
from sklearn.decomposition import PCA

def my_pca(X):
    """Performs pca on the data and returns the transformed data

    Args:
        X - the data to perform pca upon (N,D)
    
    Returns: 
        the pca components (D, D)
        explained variance (D)
    """
    N, D = np.shape(X)
    pca = PCA(n_components=D)
    pca.fit(X)
    return pca.components_, pca.explained_variance_ratio_


if __name__ == '__main__':
    from read_data import read_data
    from partition_data import partition_data
    X, y = read_data()
    X_tr, y_tr,_,_ = partition_data(X, y)
    V, variances = my_pca(X_tr)
    print 'shape of all the components (each component is a COLUMN)'
    print np.shape(V)
    import matplotlib.pyplot as plt
    plt.plot(variances)
    plt.ylabel('percentage of variance explained')
    plt.show()
    
    ## an example of getting the top 3 components
    X_pca = X.dot(V[:,0:3])
    print 'shape of data after projecting with top 3 components'
    print np.shape(X_pca)





