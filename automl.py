import numpy as np
from flaml import AutoML
from sklearn.base import BaseEstimator, clone
import warnings
warnings.simplefilter('ignore')
###################################
# AutoML models
###################################

# FLAML models don't return "self" at end of fit. We create this wrapper.


class AutoMLWrap(BaseEstimator):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def first_stage_reg(X, y):
    automl = AutoML(task='regression', time_budget=60, early_stop=True,
                    eval_method='cv', n_splits=5, metric='mse', verbose=0)
    automl.fit(X, y)
    best_est = automl.best_estimator
    return AutoMLWrap(model=clone(automl.best_model_for_estimator(best_est)))


def first_stage_clf(X, y):
    automl = AutoML(task='classification', time_budget=60, early_stop=True,
                    eval_method='cv', n_splits=5, metric='mse', verbose=0)
    automl.fit(X, y)
    best_est = automl.best_estimator
    return AutoMLWrap(model=clone(automl.best_model_for_estimator(best_est)))


# We create a custom metric to handle sample weights as we want them in RLearner (NonParamDML).
# We want to be minimizing the loss: 1/n sum_i w_i (y_i - ypred_i)^2. The standard
# mse with sample weights would have minimized (1/sum_i w_i) sum_i w_i (y_i - ypred_i)^2.
def weighted_mse(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        *args,):
    y_pred = estimator.predict(X_val)
    weight_val = 1 if weight_val is None else weight_val
    weight_train = 1 if weight_train is None else weight_train
    error = (estimator.predict(X_val) - y_val)**2
    val_loss = np.mean(weight_val * error)
    error_train = (estimator.predict(X_train) - y_train)**2
    train_loss = np.mean(weight_train * error_train)
    return val_loss, {"val_loss": val_loss, "train_loss": train_loss}

def final_stage():
    return AutoMLWrap(model=AutoML(task='regression', time_budget=60, early_stop=True,
                                   eval_method='cv',
                                   n_splits=10, metric=weighted_mse, verbose=0))