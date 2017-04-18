import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


#return the  best model and a visualization x=numTrees y = accuracy
#loop through number of trees and within loop through cv
def best_rand_forest(x, y):
	"""Computes random forest classification on data set x

    Gets the random forest classifier with the best accuracy. Tests the
    random forest accuracy using between 3 and 100 trees

    Args:
        x: a numpy array of size mxn that contains the breast cancer data set
        y: a numpy array of size m that contains M/B
        

    Returns:
        random forest classifier"""
	bestAccuracy = 0
	bestclf = None
	accuracies = []
	trees = []
	for numTrees in range(3,100):
		clf = RandomForestClassifier(n_estimators=numTrees, max_features="sqrt")
		print numTrees
		trees.append(numTrees)

		#performs cross validation on cv folds
		cvscore = cross_val_score(clf, x, y, cv = 5)
		accuracies.append(cvscore.mean())
		#Determines the best random forest classifier and its accuaracy
		if cvscore.mean() >= bestAccuracy:
			bestAccuracy = cvscore.mean()
			bestclf = clf
	#plorts the number of trees vs accuracy of the model	
	plotAccuracy(accuracies, trees)
	print bestAccuracy
	return bestclf

def plotAccuracy(accuracy, numTrees):
	fig = plt.figure(figsize=(10,4),tight_layout=True)
	ax = fig.add_subplot(1,1,1)
	plt.plot(numTrees, accuracy)
	ax.set_xlabel("Number of Trees in Forest")
	ax.set_ylabel("Accuracy")
	ax.set_title("Random Forest Accuracy", fontsize = 12)
	plt.show()

if __name__ == '__main__':
	from data_utils import get_training
	X, y = get_training()
	bestModel = best_rand_forest(X, y)
