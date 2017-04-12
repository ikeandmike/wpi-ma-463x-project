import pandas as pd

def read_data(file_name = 'complete_dataset.csv'):
    """Reads data from file

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

if __name__ == '__main__':
    X, y = read_data()
    import numpy as np
    print np.shape(X)
    print X
    print np.shape(y)
    print y
