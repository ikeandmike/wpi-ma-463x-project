import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_training, get_testing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from my_pca import my_pca

def best_logistic_regression():
  pass

if __name__=='__main__':
  # Load data
  X_train, y_train = get_training()
  N, D = X_train.shape

  # Logistic Regression
  models = [('L2, C=1', LogisticRegression()),
            ('L2, C=10', LogisticRegression(C=10)),
            ('L2, C=100', LogisticRegression(C=100)),
            ('L2, C=1000', LogisticRegression(C=1000)),
            ('L1, C=1', LogisticRegression(penalty='l1')),
            ('L1, C=10', LogisticRegression(penalty='l1', C=10)),
            ('L1, C=100', LogisticRegression(penalty='l1', C=100)),
            ('L1, C=1000', LogisticRegression(penalty='l1', C=1000))]

  print 'Without PCA'
  for name, model in models:
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
    print name, 'Val acc:', scores.mean()


  pca = PCA(n_components=D)
  X_train_reduced = pca.fit_transform(preprocessing.scale(X_train))
  variances = pca.explained_variance_ratio_

  # plt.plot(variances)
  # plt.show()


  print 'With PCA'
  for name, model in models:
    scores = cross_val_score(model, X_train_reduced[:, 0:4], y_train, scoring='accuracy', cv=10)
    print name, 'Val acc:', scores.mean()

