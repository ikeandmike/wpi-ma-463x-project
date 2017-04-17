import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Runs LDA and computes the mean accuracy on the given test data and labels.

def LDA(X_train, Y_train, X_test, Y_test):

    LDAtest = LinearDiscriminantAnalysis()
    LDAtest.fit(X_train, Y_train)

    Accuracy_train = LDAtest.score(X_train,Y_train,sample_weight=None)
    Accuracy_test = LDAtest.score(X_test,Y_test,sample_weight=None)
    
    print Accuracy_train
    print Accuracy_test

