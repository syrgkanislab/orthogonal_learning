from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from econml.policy import PolicyTree, PolicyForest
from sklearn.tree import DecisionTreeClassifier
from automl import first_stage_reg, first_stage_clf

# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# first_stage_reg = lambda X, Y: RandomForestRegressor(max_depth=5, min_samples_leaf=20)
# first_stage_clf = lambda X, Y: RandomForestClassifier(max_depth=5, min_samples_leaf=20)

final_stage = lambda: PolicyTree(max_depth=2, min_samples_leaf=20, min_balancedness_tol=.49)

def policy_dr_split(y, T, X, Xtest, n_x, cost):
    y1, y2, T1, T2, X1, X2 = train_test_split(
        y, T, X, stratify=T, train_size=.5, random_state=123)
    XT1 = np.hstack([X1, T1.reshape(-1, 1)])
    XT2 = np.hstack([X2, T2.reshape(-1, 1)])
    
    # fit nuisance models on train
    model_regression = first_stage_reg(XT1, y1)
    model_regression.fit(XT1, y1)
    model_propensity = first_stage_clf(X1, T1)
    model_propensity.fit(X1, T1)

    # predict nuisance values on test
    XT2zero = np.hstack([X2, np.zeros((X2.shape[0], 1))])
    XT2one = np.hstack([X2, np.ones((X2.shape[0], 1))])    
    regone = model_regression.predict(XT2one)
    regzero = model_regression.predict(XT2zero)
    reg = model_regression.predict(XT2)
    prop = np.clip(model_propensity.predict_proba(X2)[:, 1], .01, .99)
    reisz = (T2/prop - (1 - T2)/(1 - prop))

    # Calculate pseudo targets for policy objective sum_i theta(x_i) * pseudo_i
    rewards = np.zeros((X2.shape[0], 2))

    # doubly robust target
    pseudo = regone - regzero + reisz * (y2 - reg) - cost
    rewards[:, 1] = pseudo
    cate = final_stage().fit(X2[:, :n_x], rewards)
    
    # direct regression target
    pseudo_direct = regone - regzero - cost
    rewards[:, 1] = pseudo_direct
    cate_direct = final_stage().fit(X2[:, :n_x], rewards)
    
    # inverse propensity target
    pseudo_ips = reisz * y2 - cost
    rewards[:, 1] = pseudo_ips
    cate_ips = final_stage().fit(X2[:, :n_x], rewards)

    # return policy recommendations from each model and trained policy models
    return (cate.predict(Xtest[:, :n_x]), cate,
            cate_direct.predict(Xtest[:, :n_x]), cate_direct,
            cate_ips.predict(Xtest[:, :n_x]), cate_ips, X2, T2, reisz, reg)

def oracle_policy(y, T, X, Xtest, n_x, cost, tau_fn):
    rewards = np.zeros((X.shape[0], 2))
    rewards[:, 1] = tau_fn(X) - cost
    cate_or = final_stage().fit(X[:, :n_x], rewards)
    return cate_or.predict(Xtest[:, :n_x]), cate_or
