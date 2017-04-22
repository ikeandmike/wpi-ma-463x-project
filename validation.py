import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


class Validation(object):
    def __init__(self, X, y, k=5, bootstrap=100, normalize=True, pca=5, lasso=-1.0):
        """
            Validates a model. Performs (optionally) bootstrap, cross val, 
            normalization, pca and then lasso. In that order. 

            Args:
                k - if greater than 0, performs CV with k folds
                bootstrap - if greater than 0, creates bootstrapped samples 
                normalize - znormalizes the data if True
                pca - if greater than 0, performs pca and keeps that many components
                lasso - if greater than 0, performs lasso with alpha
            
            Returns:
                A validation object that can fit a model 
                Ex:
                | val = Validation(X_tr, y_tr, 5, -1, True, 10);
                | results = val.fit(model)
                
        """
        self.X = X
        self.y = y
        self.normalize = normalize

        self.k = k

        self.bootstrap = bootstrap > 0
        self.samples = bootstrap

        self.pca = pca > 0
        self.components = pca

        self.lasso = lasso > 0
        self.alpha = lasso

        self.results = {
            'accuracy' : [],
            'recall': [],
            'precision': []
        }

    def update(self, k=5, bootstrap=100, normalize=True, pca=5, lasso=-1.0):
        """ updates validation framework with new hyper-params (preprocessing)

        """
        self.normalize = normalize

        self.k = k

        self.bootstrap = bootstrap > 0
        self.samples = bootstrap

        self.pca = pca > 0
        self.components = pca

        self.lasso = lasso > 0
        self.alpha = lasso

    def cross_val_accuracy(self, model):
        """
            performs validation on model, and returns the results
            general use will be to change the model and test it

            Args:
                model - the model to validate

            Returns:
                a dictionary of lists of accuracy, recall and precision
                each being a scalar. Use get_detailed_results to get 
                lists of each metric.
        """
        aggregates = []
        N = len(self.y)
        if self.bootstrap:
            for i in xrange(self.samples):
                ind = np.random.choice(N, N, True)
                aggregates.append([self.X[ind,:], self.y[ind]])
        else:
            aggregates.append([self.X, self.y])

        # for each sample (or just X,y)
        K = int(N/(self.k * 1.0))
        for X, y in aggregates:
            for k in xrange(self.k):
                # split sample
                ind = np.ones(N)
                ind[k*K:(k+1)*K] = 0
                X_tr = X[(ind==1),:]
                y_tr = y[(ind==1)]
                X_te = X[(ind==0),:]
                y_te = y[(ind==0)]

                # normalize
                if self.normalize:
                    # scale on training
                    s = StandardScaler()
                    X_tr = s.fit_transform(X_tr);
                    # scale test using train params
                    X_te = s.transform(X_te);

                # pca
                if self.pca:
                    pca = PCA(n_components=self.components)
                    # find components on train
                    X_tr = pca.fit_transform(X_tr) #[:,0:self.components]
                    # transform test on train
                    X_te = pca.transform(X_te) #[:,0:self.components]
                    
                # lasso
                if self.lasso:
                    clf = Lasso(alpha=self.alpha)
                    # perform lasso
                    clf.fit(X_tr, y_tr)
                    coef = clf.coef_
                    keep = coef > 0.0
                    X_tr = X_tr[:,keep]
                    X_te = X_te[:,keep]

                # train & test
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_te)
                self.results['accuracy'].append(accuracy_score(y_te, y_hat))
                self.results['recall'].append(recall_score(y_te, y_hat))
                self.results['precision'].append(precision_score(y_te, y_hat))
            # end cv
        # end bootstrapping
        return np.array(self.results['accuracy']).mean()

    def get_detailed_results(self):
        """Helper function that gets full set of results

            Returns:
                Returns a dictionary of results as follows:
                {
                    'accuracy' : [list of accuracies],
                    'recall': [list of recall results],
                    'precision': [list of precision results]
                }
        """
        return self.results

if __name__ == '__main__':
    from data_utils import get_training
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = get_training()
    mdl = LinearDiscriminantAnalysis()
    val = Validation(X, y)

    pca_comp_to_test = [ 1, 3, 5, 10, 30]
    results = []
    best_so_far = (0, -1)

    # common validation pattern
    for d in pca_comp_to_test:
        # update hyper params for preprocessing
        val.update(pca=d)
        
        # generally change model params
        acc = val.cross_val_accuracy(mdl)
        results.append(acc)
        # if you're interested in more details, use val.get_detailed_results()

        # update best so far
        if acc > best_so_far[0]:
            best_so_far = (acc, d)

    # analyze results...
    print 'best # of comp for pca for LDA: {0} (accuracy: {1})'.format(best_so_far[1], best_so_far[0])
