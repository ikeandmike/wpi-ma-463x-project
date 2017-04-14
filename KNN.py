from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

# Runs KNN and computes the accuracy of the model using cross validation with 10 folds by default.
# Returns a string with the accuracy +/- 2*std

# Warning
# Regarding the Nearest Neighbors algorithms, if it is found that two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data. 

def knn(X, Y, k, numCVFolds=10):
	myKNN = KNeighborsClassifier(n_neighbors=k)
	cvscore = cross_val_score(myKNN, X, Y, cv = numCVFolds)
	
	accuracy = ("Accuracy: %0.2f (+/- %0.2f)" % (cvscore.mean(), cvscore.std() * 2))
	print accuracy
	return accuracy
	
