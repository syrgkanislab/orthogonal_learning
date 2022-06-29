import numpy as np
from sklearn.base import BaseEstimator
from slearner import MySLearner
from econml.dr import DRLearner
from sklearn.model_selection import train_test_split
from econml.dml import NonParamDML
from automl import first_stage_clf, first_stage_reg, final_stage

# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# first_stage_reg = lambda X, Y: RandomForestRegressor(max_depth=5, min_samples_leaf=20)
# first_stage_clf = lambda X, Y: RandomForestClassifier(max_depth=5, min_samples_leaf=20)
# final_stage = lambda: RandomForestRegressor(max_depth=4, min_samples_leaf=20)

####################################
# methods
####################################

class OracleY(BaseEstimator):
    def __init__(self, *, base_fn, tau_fn, prop_fn):
        self.base_fn = base_fn
        self.tau_fn = tau_fn
        self.prop_fn = prop_fn

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.base_fn(X) + self.tau_fn(X) * (self.prop_fn(X) - .5)


class OracleT(BaseEstimator):
    def __init__(self, *, prop_fn):
        self.prop_fn = prop_fn

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        prob = self.prop_fn(X).reshape(-1, 1)
        return np.hstack([1 - prob, prob])


def oracle_gen(base_fn, tau_fn, prop_fn):
    def oracle(y, T, X, Xtest, n_x):
        est = NonParamDML(model_y=OracleY(base_fn=base_fn, tau_fn=tau_fn, prop_fn=prop_fn),
                          model_t=OracleT(prop_fn=prop_fn),
                          model_final=final_stage(),
                          discrete_treatment=True,
                          cv=1,
                          random_state=123)
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est
    return oracle


def dml(y, T, X, Xtest, n_x):
    model_y = first_stage_reg(X, y)
    print(model_y)
    model_t = first_stage_clf(X, T)
    print(model_t)
    est = NonParamDML(model_y=model_y,
                      model_t=model_t,
                      model_final=final_stage(),
                      discrete_treatment=True,
                      cv=10,
                      random_state=123)
    est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
    return est.effect(Xtest[:, :n_x]), est


def dml_split(y, T, X, Xtest, n_x):
    y1, y2, T1, T2, X1, X2 = train_test_split(
        y, T, X, stratify=T, train_size=.5, random_state=123)
    model_y = first_stage_reg(X1, y1)
    model_y.fit(X1, y1)
    yres = y2 - model_y.predict(X2)
    model_t = first_stage_clf(X1, T1)
    model_t.fit(X1, T1)
    # to avoid division by zero
    Tres = T2 - np.clip(model_t.predict_proba(X2)[:, 1], .001, .999)
    cate = final_stage()
    cate.fit(X2[:, :n_x], yres/Tres, sample_weight=Tres**2)
    return cate.predict(Xtest[:, :n_x]), cate


def dr(y, T, X, Xtest, n_x):
    model_regression = first_stage_reg(np.hstack([X, T.reshape(-1, 1)]), y)
    print(model_regression)
    model_propensity = first_stage_clf(X, T)
    print(model_propensity)
    dr = DRLearner(model_regression=model_regression,
                   model_propensity=model_propensity,
                   model_final=final_stage(),
                   min_propensity=.1,
                   cv=10, random_state=123)
    dr.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
    return dr.effect(Xtest[:, :n_x]), dr


def dr_split(y, T, X, Xtest, n_x):
    y1, y2, T1, T2, X1, X2 = train_test_split(
        y, T, X, stratify=T, train_size=.5, random_state=123)
    XT1 = np.hstack([X1, T1.reshape(-1, 1)])
    XT2 = np.hstack([X2, T2.reshape(-1, 1)])
    model_regression = first_stage_reg(XT1, y1)
    model_regression.fit(XT1, y1)
    model_propensity = first_stage_clf(X1, T1)
    model_propensity.fit(X1, T1)

    XT2zero = np.hstack([X2, np.zeros((X2.shape[0], 1))])
    XT2one = np.hstack([X2, np.ones((X2.shape[0], 1))])
    pseudo = model_regression.predict(
        XT2one) - model_regression.predict(XT2zero)
    prop = np.clip(model_propensity.predict_proba(X2)[:, 1], .1, .9)
    reisz = (T2/prop - (1 - T2)/(1 - prop))
    pseudo += reisz * (y2 - model_regression.predict(XT2))

    cate = final_stage()
    cate.fit(X2[:, :n_x], pseudo)
    return cate.predict(Xtest[:, :n_x]), cate


def myslearner(y, T, X, Xtest, n_x):
    slearner = MySLearner(overall_model=first_stage_reg(np.hstack([T.reshape(-1, 1), X]), y),
                          final_model=final_stage())
    slearner.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
    return slearner.effect(Xtest[:, :n_x]), slearner
