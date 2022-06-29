# orthogonal_learning
Experiments for the paper: Orthogonal Statistical Learning

To generate all paper tables run the jupyter notebook `RunAllExperiments.ipynb`

The code requires the following python packages:
- `flaml`: https://github.com/microsoft/FLAML
- `econml`: https://github.com/microsoft/EconML
- `scikit-learn`: https://github.com/scikit-learn/scikit-learn
- `numpy`: https://numpy.org/

The code is organized as follows:
- `experiments.py`: Contains all the logic for generating data for different setups, running an experiment and storing the result.
- `automl.py`: contains wrappers for nuisance and target automl FLAML models
- `cate.py`: contains implementations of all the CATE estimation methods
- `policy.py`: contains implmentations of all the Policy learning methods
- `slearner.py`: contains an SLearner cate estimator class that is used in `cate.py`.