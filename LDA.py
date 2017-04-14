import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# Runs LDA and computes the accuracy of the model using cross validation with 10 folds by default.
# Returns a string with the accuracy +/- 2*std


def LDA(X_train, Y_train, numCVFolds=10):

    LDAtest = LinearDiscriminantAnalysis()
    LDAtest.fit(X_train, Y_train)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)

    cvscore = cross_val_score(LDAtest, X_train, Y_train, cv=numCVFolds)

    accuracy = ("Accuracy: %0.2f (+/- %0.2f)" % (cvscore.mean(), cvscore.std() * 2))
    print accuracy
    return accuracy

