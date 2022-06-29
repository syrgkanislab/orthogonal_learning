import numpy as np
from sklearn.base import clone
from econml._cate_estimator import LinearCateEstimator

class MySLearner(LinearCateEstimator):
    
    def __init__(self, *, overall_model, final_model):
        self.overall_model = overall_model
        self.final_model = final_model
        return 

    def fit(self, y, T, X, W=None):
        XW = X
        if W is not None:
            XW = np.hstack([X, W])
        self.model_ = clone(self.overall_model)
        self.model_.fit(np.hstack([T.reshape(-1, 1), X]), y)
        ones = np.hstack([np.ones((X.shape[0], 1)), X])
        zeros = np.hstack([np.zeros((X.shape[0], 1)), X])
        diffs = self.model_.predict(ones) - self.model_.predict(zeros)
        
        self.model_final_ = clone(self.final_model)
        self.model_final_.fit(X, diffs)
        return self

    def effect(self, X, T0=0, T1=1):
        return self.const_marginal_effect(X)

    def const_marginal_effect(self, X):
        return self.model_final_.predict(X)