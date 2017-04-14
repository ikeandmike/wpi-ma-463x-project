import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
	from data_utils import get_training
	X, y = get_training()
	accuracy = randForest(X, y, 6, 10)
