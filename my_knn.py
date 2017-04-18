from data_utils import * # Utility functions we wrote

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

"""
Findings

Model performs around the same for K between 3 and 21. This implies that the classes of data are very well-separable.
"""

# Warning
# Regarding the Nearest Neighbors algorithms, if it is found that two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data. 

def best_knn():
	return KNeighborsClassifier(n_neighbors=3) # Empirically, k=3 was best under both 10-fold and LOO CV.

def getScores(X, Y, k_list, num_cv_folds):
	kfold = KFold(n_splits=num_cv_folds)

	scores = []
	for k in k_list:
		print("Evaluating KNN with k=%2d" % k)
		my_knn = KNeighborsClassifier(n_neighbors=k)
		scores.append(cross_val_score(my_knn, X, Y, cv = kfold).mean())

	return scores

if __name__ == '__main__':

	X, Y = get_training()

	k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 31, 51, 71, 91]

	ten_scores = getScores(X, Y, k_list, 10)
	loocv_scores = getScores(X, Y, k_list, len(X))

	print("____________________________________________________________")
	for i in range(len(k_list)):
		k = k_list[i]
		acc1 = ten_scores[i]
		acc2 = loocv_scores[i]
		print("| k = %2d | 10-Fold Accuracy: %.3f | LOOCV Accuracy: %.3f |" % (k, acc1, acc2))
	print("____________________________________________________________")

	# TODO Generate graphs of k vs. accuracy
