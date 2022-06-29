import os
from joblib import Parallel, delayed
import argparse
import numpy as np
import joblib
from policy import policy_dr_split, oracle_policy
from cate import oracle_gen, dml, dml_split, dr, dr_split, myslearner
import warnings
warnings.simplefilter('ignore')


####################################
# utilities
####################################
def rmse(ytrue, y):
    return np.sqrt(np.mean((ytrue.flatten() - y.flatten())**2))

####################################
# data gen
####################################


def gen_data(n, d, base_fn, tau_fn, prop_fn, sigma, dist='uniform'):
    if dist == 'uniform':
        X = np.random.uniform(0, 1, size=(n, d))
        Xtest = np.random.uniform(0, 1, size=(10000, d))
    if dist == 'normal':
        X = np.random.normal(0, 1, size=(n, d))
        Xtest = np.random.normal(0, 1, size=(10000, d))
    if dist == 'centered_uniform':
        X = np.random.uniform(-.5, .5, size=(n, d))
        Xtest = np.random.uniform(-.5, .5, size=(10000, d))
    T = np.random.binomial(1, prop_fn(X))
    y = (T - .5) * tau_fn(X) + base_fn(X) + \
        sigma * np.random.normal(0, 1, size=(n,))
    return y, T, X, Xtest


def get_data_generator(setup, n, d, sigma):
    
    if setup == 'A':
        dist = 'uniform'
        def base_fn(X): return np.sin(
            np.pi * X[:, 0] * X[:, 1]) + 2*(X[:, 2] - .5)**2 + X[:, 3] + .5*X[:, 4]
        def prop_fn(X): return np.clip(
            np.sin(np.pi * X[:, 0] * X[:, 1]), .2, .8)

        def tau_fn(X): return .2 + (X[:, 0] + X[:, 1]) / 2
    elif setup == 'B':
        dist = 'centered_uniform'
        def base_fn(X): return np.maximum(0, np.maximum(
            X[:, 0] + X[:, 1], X[:, 2])) + np.maximum(X[:, 3] + X[:, 4], 0)

        def prop_fn(X): return .5 * np.ones(X.shape[0])
        def tau_fn(X): return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    elif setup == 'C':
        dist = 'centered_uniform'
        def base_fn(X): return 2 * np.log(1 + np.exp(np.sum(X[:, :3], axis=1)))
        def prop_fn(X): return 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
        def tau_fn(X): return np.ones(X.shape[0])
    elif setup == 'D':
        dist = 'centered_uniform'
        def base_fn(X): return .5 * (np.maximum(0, np.sum(X[:, :3], axis=1)) + np.maximum(0, X[:, 3] + X[:, 4]))
        def prop_fn(X): return 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
        def tau_fn(X): return np.maximum(0, np.sum(X[:, :3], axis=1)) - np.maximum(0, X[:, 3] + X[:, 4])
    elif setup == 'E':
        dist = 'centered_uniform'
        def base_fn(X): return 5 * np.maximum(0, X[:, 0] + X[:, 1])
        def prop_fn(X): return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))
        def tau_fn(X): return 2 * ((X[:, 0] > 0.1) | (X[:, 1] > 0.1)) - 1
    elif setup == 'F':
        dist = 'centered_uniform'
        def base_fn(X): return 5 * np.maximum(0, X[:, 0] + X[:, 1])
        def prop_fn(X): return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))
        def tau_fn(X): return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    else:
        raise AttributeError(f"Invalid parameter setup={setup}")

    def gen_data_fn(): return gen_data(
        n, d, base_fn, tau_fn, prop_fn, sigma, dist=dist)
    
    return gen_data_fn, base_fn, tau_fn, prop_fn

####################################
# experiment definition
####################################

def exp(data, method_list, tau_fn, n_x):
    y, T, X, Xtest = data
    results = {}
    for name, method in method_list:
        tau_est, est = method(y, T, X, Xtest, n_x)
        results[name] = rmse(tau_fn(Xtest), tau_est)**2
    return results

def policy_exp(data, base_fn, tau_fn, prop_fn, n_x, cost, plot=False):
    y, T, X, Xtest = data
    pred_or, est_or = oracle_policy(y, T, X, Xtest, n_x, cost, tau_fn)
    (pred_dr, est_dr, pred_direct, est_direct, pred_ips, est_ips,
     X2, T2, reisz, reg) = policy_dr_split(y, T, X, Xtest, n_x, cost)
    opt = np.mean((tau_fn(Xtest) - cost) * (tau_fn(Xtest) - cost > 0))
    hat_or = np.mean((tau_fn(Xtest) - cost) * (pred_or > 0))
    hat_dr = np.mean((tau_fn(Xtest) - cost) * (pred_dr > 0))
    hat_direct = np.mean((tau_fn(Xtest) - cost) * (pred_direct > 0))
    hat_ips = np.mean((tau_fn(Xtest) - cost) * (pred_ips > 0))
    true_reisz = T2/prop_fn(X2) - (1 - T2) / (1 - prop_fn(X2))
    true_reg = (T2 - .5) * tau_fn(X2) + base_fn(X2)
    return {'opt': opt, 'or': hat_or, 'dr': hat_dr, 
            'direct': hat_direct, 'ips': hat_ips,
            'reisz_rmse': np.sqrt(np.mean(np.mean((true_reisz - reisz)**2))),
            'reg_rmse': np.sqrt(np.mean(np.mean((true_reg - reg)**2)))}


def main(setup, n, d, n_x, sigma, start_sample, sample_its, target_dir, policy=False):
    np.random.seed(123)

    gen_data_fn, base_fn, tau_fn, prop_fn = get_data_generator(setup, n, d, sigma)

    samples = [gen_data_fn() for _ in range(100)]

    if policy:
        y, T, X, Xtest = gen_data_fn()
        cost = np.mean(tau_fn(Xtest))
        res = Parallel(n_jobs=-1, verbose=3)(delayed(policy_exp)(samples[it], base_fn, tau_fn, prop_fn, n_x, cost)
                                             for it in np.arange(start_sample, start_sample + sample_its))
        fname = 'res_policy.jbl'
    else:
        method_list = [('oracle', oracle_gen(base_fn, tau_fn, prop_fn)),
                    ('dml', dml),
                    ('dml_split', dml_split),
                    ('dr', dr), ('dr_split', dr_split),
                    ('myslearner', myslearner)]

        res = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(samples[it], method_list, tau_fn, n_x)
                                            for it in np.arange(start_sample, start_sample + sample_its))
        fname = 'res.jbl'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    joblib.dump(res, os.path.join(target_dir, fname))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-setup', type=str, required=False, default='A',
                        help='The dgp setup')
    parser.add_argument('-n', type=int, required=True, help='N samples')
    parser.add_argument('-d', type=int, required=True, help='N controls')
    parser.add_argument('-n_x', type=int, required=True,
                        help='N hetero features')
    parser.add_argument('-sigma', type=float, required=True,
                        help='Scale of outcome noise')
    parser.add_argument('-start_sample', type=int,
                        required=True, help='Start experiment')
    parser.add_argument('-sample_its', type=int,
                        required=True, help='Number of experiments')
    parser.add_argument('-dir', type=str, required=False, default=os.environ['AMLT_OUTPUT_DIR'],
                        help='directory for outputs')
    parser.add_argument('-policy', type=int, required=False, default=0,
                        help='is it a policy experiment')
    args = parser.parse_args()
    main(args.setup, args.n, args.d, args.n_x, args.sigma,
         args.start_sample, args.sample_its, args.dir, policy=args.policy)
