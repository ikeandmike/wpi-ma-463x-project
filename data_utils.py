import pandas as pd
import numpy as np


_PREDICTORS = ['radius', 'texture', 'perimeter', 'area', 'smooth',
               'compact', 'concav', 'concavpt', 'sym', 'frac']
HEADER = ['diag']                                 +\
          ['mean_' + pred for pred in _PREDICTORS] +\
          ['std_' + pred for pred in _PREDICTORS]  +\
          ['worst_' + pred for pred in _PREDICTORS]


def get_training(as_dataframe=False):
    """partitions data into final train

    Args:
        as_dataframe: set this to True to return labels and predictors in a
                      DataFrame with named columns
    Returns:
        The training data
        X_tr, y_tr - the data and class labels for training
    """
    X, y = _read_data()
    X_tr, y_tr,_,_ = _partition_data(X, y)

    if as_dataframe:
        combined = np.hstack((y_tr.reshape(y_tr.size, 1), X_tr))
        combined = pd.DataFrame(combined, columns=HEADER)
        return combined
    else:
        return X_tr, y_tr

def get_testing(as_dataframe=False):
    """partitions data into final test

    Args:
        as_dataframe: set this to True to return labels and predictors in a
                      DataFrame with named columns
    Returns:
        The testing data
        X_te, y_te - the data and class labels for testing
    """
    X, y = _read_data()
    _, _, X_te, y_te = _partition_data(X, y)

    if as_dataframe:
        combined = np.hstack((y_te.reshape(y_te.size, 1), X_te))
        combined = pd.DataFrame(combined, columns=HEADER)
        return combined
    else:
        return X_te, y_te

def _read_data(file_name='complete_dataset.csv'):
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
