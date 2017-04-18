import pandas as pd
import numpy as np
from data_utils import get_training, get_testing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from my_pca import my_pca

def best_logistic_regression():
  pass

if __name__=='__main__':
  # Load data
  X_train, y_train = get_training()

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


  print 'With PCA'
  X_train_reduced = my_pca(X_train, 3)
  for name, model in models:
    scores = cross_val_score(model, X_train_reduced, y_train, scoring='accuracy', cv=10)
    print name, 'Val acc:', scores.mean()

