from data_utils import get_training
from my_pca import my_pca
from my_lasso import my_lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def best_lda():
    #TODO find best LDA using data
    pass

def generate_subsets():
    """Returns some of the subsets of predictors
    """
    out = [ range(30),
            [0,1,2,3,4,5],
            [5,6,7,8,9,10],
            [1,2,3],
            [21,23,5,6,2],
            [4,19,12,5,6]]
    return out

def getScores(X, y, subsets, num_cv_folds):
    kFold = KFold(n_splits=num_cv_folds)
    subsets = [X[:, subset] for subset in subsets]
    scores = []
    my_lda = LinearDiscriminantAnalysis()
    for subset in subsets:
        print "Evaluating LDA with a subset of predictors"
        my_lda.fit(subset, Y)
        scores.append(cross_val_score(my_lda, subset, Y, cv = kFold).mean())
    return scores

def getScores_pca(X, Y, num_pred_list, num_cv_folds):
    kfold = KFold(n_splits=num_cv_folds)
    
    scores = []
    for p in num_pred_list:
        print("Evaluating LDA with predictors=%2d" % p)
        my_LDA = LinearDiscriminantAnalysis()
        my_LDA.fit(my_pca(X, p), Y)
	scores.append(cross_val_score(my_LDA, X, Y, cv = kfold).mean())

    return scores

def getScores_lasso(X, Y, alpha_list, num_cv_folds):
    kfold = KFold(n_splits=num_cv_folds)
    
    scores = []
    for a in alpha_list:
        print("Evaluating LDA with alpha=%2d" % a)
        my_LDA = LinearDiscriminantAnalysis()
        my_LDA.fit(my_lasso(X, Y, a), Y)
	scores.append(cross_val_score(my_LDA, X, Y, cv = kfold).mean())

    return scores

def plotAccuracy(accuracy, pred, title):
    fig = plt.figure(figsize=(10,4),tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    plt.plot(k, accuracy)
    ax.set_xlabel("Predictors")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize = 12)
    plt.show()

if __name__ == '__main__':

    X, Y = get_training()

    num_pred_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 30]
    alpha_list = [0,.125, .25, .375, .5, .625, .75, .875, 1]
    subsets = generate_subsets()

    ten_scores_subsets = getScores(X, Y, subsets, 10)
    loocv_scores_subsets = getScores(X, Y, subsets, len(X))
 
    ten_scores_pca = getScores_pca(X, Y, num_pred_list, 10)
    loocv_scores_pca = getScores_pca(X, Y, num_pred_list, len(X))
    ten_scores_lasso = getScores_lasso(X, Y, alpha_list, 10)
    loocv_scores_lasso = getScores_lasso(X, Y, alpha_list, len(X))

    print("_______________________Subsets______________________________")
    for i in range(len(subsets)):
	acc1 = ten_scores_subsets[i]
	acc2 = loocv_scores_subsets[i]
	print("| subset = %2d | 10-Fold Accuracy: %.3f | LOOCV Accuracy: %.3f |" % (i, acc1, acc2))
    plotAccuracy(ten_scores, subsets, "(10 Fold CV)")
    plotAccuracy(loocv_scores, subsets, "(LOOCV)")
    print("____________________________________________________________")


    print("_______________________PCA__________________________________")
    for i in range(len(num_pred_list)):
	p = num_pred_list[i]
	acc1 = ten_scores_pca[i]
	acc2 = loocv_scores_pca[i]
	print("| predictors = %2d | 10-Fold Accuracy: %.3f | LOOCV Accuracy: %.3f |" % (p, acc1, acc2))
    plotAccuracy(ten_scores, num_pred_list, "(10 Fold CV)")
    plotAccuracy(loocv_scores, num_pred_list, "(LOOCV)")
    print("____________________________________________________________")

    print("_______________________Lasso________________________________")
    for i in range(len(alpha_list)):
	a = alpha_list[i]
	acc1 = ten_scores_lasso[i]
	acc2 = loocv_scores_lasso[i]
	print("| alpha = %.3f | 10-Fold Accuracy: %.3f | LOOCV Accuracy: %.3f |" % (a, acc1, acc2))
    plotAccuracy(ten_scores, alpha_list, "(10 Fold CV)")
    plotAccuracy(loocv_scores, alpha_list, "(LOOCV)")
    print("____________________________________________________________")
    # TODO Generate graphs of num_pred vs. accuracy


"""
output:
    _______________________Subsets______________________________
    | subset =  0 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | subset =  1 | 10-Fold Accuracy: 0.902 | LOOCV Accuracy: 0.914 |
    | subset =  2 | 10-Fold Accuracy: 0.906 | LOOCV Accuracy: 0.908 |
    | subset =  3 | 10-Fold Accuracy: 0.885 | LOOCV Accuracy: 0.902 |
    | subset =  4 | 10-Fold Accuracy: 0.921 | LOOCV Accuracy: 0.929 |
    | subset =  5 | 10-Fold Accuracy: 0.894 | LOOCV Accuracy: 0.900 |
    ____________________________________________________________
    _______________________PCA__________________________________
    | predictors =  3 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors =  5 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors =  7 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors =  9 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 11 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 13 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 15 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 17 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 19 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 21 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | predictors = 30 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    ____________________________________________________________
    _______________________Lasso________________________________
    | alpha = 0.000 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.125 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.250 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.375 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.500 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.625 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.750 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 0.875 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    | alpha = 1.000 | 10-Fold Accuracy: 0.964 | LOOCV Accuracy: 0.960 |
    ____________________________________________________________
"""
