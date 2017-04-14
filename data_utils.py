import pandas as pd
import numpy as np 

def get_training():
    """partitions data into final train

    Returns:
        The train
        X_tr, y_tr - the data and class labels for training
    """
    X, y = _read_data()
    X_tr, y_tr,_,_ = _partition_data(X, y)
    return X_tr, y_tr

def get_testing():
    """partitions data into final test

    Returns:
        The testing data
        X_te, y_te - the data and class labels for testing
    """
    X, y = _read_data()
    _, _, X_te, y_te = _partition_data(X, y)
    return X_te, y_te

def _read_data(file_name = 'complete_dataset.csv'):
    """Reads data from file (private function)

    Gets the data from the csv and splits it into X,y
    This assumes the breast cancer formatting and predictors

    Args:
        file_name: (optional) the file to read

    Returns:
        X - an numpy array of size (N,D) containing all the data in the set
        y - a numpy array of size (N,) containing class labels
    """
    
    df=pd.read_csv(file_name, sep=',',header=None)
    values = df.values
    y = values[:,1]
    X = values[:,2:]
    return X, y

def _partition_data(X, y, p=0.85):
    """partitions data into final train and test (private function)

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
    X, y = _read_data()
    import numpy as np
    print np.shape(X)
    print X
    print np.shape(y)
    print y

    X, y = _read_data()
    X_tr, y_tr, X_te, y_te = _partition_data(X,y)
    print np.shape(X_tr), np.shape(y_tr)
    print np.shape(X_te), np.shape(y_te)

    get_training()
    get_testing()
