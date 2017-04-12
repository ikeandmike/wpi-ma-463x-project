import numpy as np

def partition_data(X, y, p=0.85):
    """partitions data into final train and test 

    Args:
        X - the training data (N, D)
        y - the labels (N,)
        p - the percentage in train

    Returns:
        The train and test data
        X_tr, y_tr - the data and class labels for training
        X_te, y_te - the data and class labels for testing
    """
    np.random.seed(1)
    keep = np.random.rand(np.shape(y)[0]) < p
    X_tr = X[keep,:]
    y_tr = y[keep]
    X_te = X[~keep,:]
    y_te = y[~keep]

    return X_tr, y_tr, X_te, y_te

if __name__ == '__main__':
    from read_data import read_data
    X, y = read_data()
    X_tr, y_tr, X_te, y_te = partition_data(X,y)
    print np.shape(X_tr), np.shape(y_tr)
    print np.shape(X_te), np.shape(y_te)
