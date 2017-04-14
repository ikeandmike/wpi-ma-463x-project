from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np

# Runs QDA and computes the mean accuracy on the given data and labels 


def QDA(X_train, Y_train, X_test, Y_test):

    QDAtest = QuadraticDiscriminantAnalysis()
    QDAtest.fit(X_train, Y_train)
    QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=False, tol=0.0001)

    Accuracy_train = QDAtest.score(X_train,Y_train,sample_weight=None)
    Accuracy_test = QDAtest.score(X_test,Y_test,sample_weight=None)
    
    print Accuracy_train
    print Accuracy_test
