from data_utils import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Runs QDA and computes the mean accuracy on the given data and labels 


def QDA(X_train, Y_train, X_test, Y_test):

    QDAtest = QuadraticDiscriminantAnalysis()
    QDAtest.fit(X_train, Y_train)

    Accuracy_train = QDAtest.score(X_train,Y_train,sample_weight=None)
    Accuracy_test = QDAtest.score(X_test,Y_test,sample_weight=None)
    
    print Accuracy_train
    print Accuracy_test
