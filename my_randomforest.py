import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bootstrap import bootstrap

def randForest(x, y, numTrees, cv):
	"""Computes random forest classification on data set x

    Gets the random forest classifier, assumes want 100 random 
    trees, the number of features of each tree to be sqrt of the
    total features.

    Args:
        x: a numpy array of size mxn that contains the breast cancer data set
        y: a numpy array of size m that contains M/B
        numTrees: number of trees wanted in the random forest
        cv: number of k-folds wanted for the cross validation

    Returns:
        accuracy: a string giving the cross validation accuracy
    """
	
	clf = RandomForestClassifier(n_estimators=numTrees, max_features="sqrt")
	#performs cross validation on cv folds
	cvscore = cross_val_score(clf, x, y, cv = cv)
	accuracy = ("Accuracy: %0.2f (+/- %0.2f)" % (cvscore.mean(), cvscore.std() * 2))
	print accuracy
	return accuracy

if __name__ == '__main__':
	from read_data import read_data
	from partition_data import partition_data
	X, y = read_data()
	print "Shape of X: " + str(X.shape)
	print "Shape of Y: " + str(y.shape)
	xtrain, ytrain,xtest,ytest = partition_data(X, y)
	print "Shape of Xtrain: " + str(xtrain.shape)
	print "Shape of Ytrain: " + str(ytrain.shape)
	accuracy = randForest(xtrain, ytrain, 6, 10)