from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np

# Runs QDA and computes the accuracy of the model using cross validation with 10 folds by default.
# Returns a string with the accuracy +/- 2*std


def QDA(X_train, Y_train, numCVFolds=10):

    QDAtest = QuadraticDiscriminantAnalysis()
    QDAtest.fit(X_train, Y_train)
    QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=False, tol=0.0001)

    cvscore = cross_val_score(QDAtest, X_train, Y_train, cv=numCVFolds)

    accuracy = ("Accuracy: %0.2f (+/- %0.2f)" % (cvscore.mean(), cvscore.std() * 2))
    print accuracy
    return accuracy
