from data_utils import *
from my_pca import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def best_QDA():
    #TODO find best QDA using data

def getScores(X, Y, num_pred_list, num_cv_folds):
	kfold = KFold(n_splits=num_cv_folds)
    
	scores = []
	for p in num_pred_list:
		print("Evaluating QDA with predictors=%2d" % p)
		my_QDA = QuadraticDiscriminantAnalysis()
        my_QDA.fit(my_pca(X, p), Y)
		scores.append(cross_val_score(my_QDA, X, Y, cv = kfold).mean())

	return scores

if __name__ == '__main__':

	X, Y = get_training()

	num_pred_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 30]

	ten_scores = getScores(X, Y, num_pred_list, 10)
	loocv_scores = getScores(X, Y, num_pred_list, len(X))

	print("____________________________________________________________")
	for i in range(len(num_pred_list)):
		p = num_pred_list[i]
		acc1 = ten_scores[i]
		acc2 = loocv_scores[i]
		print("| predictors = %2d | 10-Fold Accuracy: %.3f | LOOCV Accuracy: %.3f |" % (p, acc1, acc2))
	print("____________________________________________________________")

	# TODO Generate graphs of num_pred vs. accuracy
