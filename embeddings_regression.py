import functools
from typing import Tuple

import h5py
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch
from matplotlib import cm
from pyro.distributions.transforms import neural_autoregressive

from pyro.infer import Predictive, NUTS, MCMC, SVI, Trace_ELBO,JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import pyro.distributions as dist
from pyro import poutine
from pyro.nn import PyroSample, PyroModule
from ray import tune
from ray.air import ScalingConfig, RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn import manifold, linear_model, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights

from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.utils import check_array
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

import xgboost as xgb

from modules import CNN
from settings import data_dir, result_dir, log_dir
from loaders import H5Dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cuda'
embedding_experiment="VAE-tuned-flow"
experiment="Xgboost-embeddings"

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))
embeddings=joblib.load(os.path.join(result_dir,f"Embeddings-{embedding_experiment}.joblib"))


Train_embeddings=np.concatenate([train for train,sub in embeddings],axis=1)
Submission_embeddings=np.concatenate([sub for train,sub in embeddings],axis=1)






def r2_score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    r2_=metrics.r2_score(y,predt)
    return 'R2', r2_



def get_best_model_checkpoint(results):
    best_bst = xgb.Booster()
    best_result = results.get_best_result()

    with best_result.checkpoint.as_directory() as best_checkpoint_dir:
        best_bst.load_model(os.path.join(best_checkpoint_dir, "model.xgb"))
    rmse= best_result.metrics["eval-rmse"]
    r2=best_result.metrics['eval-R2']
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model RMSE: {rmse:.4f} | R2: {r2:.2f}")
    return best_bst

search_space = {
    # You can mix constants with search space objects.
    "objective": "reg:squarederror",
    "eval_metric": ["logloss", "rmse"],
    "max_depth": tune.randint(1, 50),
    "min_child_weight": tune.choice([1, 2, 3]),
    "subsample": tune.uniform(0.5, 1.0),
    "eta": tune.loguniform(1e-4, 1.0),


}


cluster_predictions=[]
cluster_state_config=[]
for c in Train.cluster.unique():
    # train_data = embeddings[c][0][Train['cluster'] != c, :]
    # val_data = embeddings[c][0][Train['cluster'] == c, :]
    # submission_data=embeddings[c][1]
    #
    # train_y = Train.loc[Train['cluster'] != c, 'biomass'].values
    # val_y = Train.loc[Train['cluster'] == c, 'biomass'].values

    train_data,val_data,train_y,val_y=train_test_split(Train_embeddings,Train['biomass'].values,stratify=Train['cluster'],random_state=c)
    submission_data=Submission_embeddings


    # scl = StandardScaler()
    # train_data = scl.fit_transform(train_data)
    # val_data = scl.transform(val_data)
    # submission_data=scl.transform(submission_data)



    scl_biomass = QuantileTransformer(output_distribution='normal', n_quantiles=100)
    train_y=scl_biomass.fit_transform(train_y.reshape(-1,1)).reshape(-1)
    val_y=scl_biomass.transform(val_y.reshape(-1,1)).reshape(-1)





    def trainer(config: dict):
        train_set = xgb.DMatrix(train_data, label=train_y)
        test_set = xgb.DMatrix(val_data, label=val_y)
        xgb.train(
            config,
            train_set,
            evals=[(test_set, "eval")],
            verbose_eval=False,
            custom_metric=r2_score,
            num_boost_round=150,
            callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
        )
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=150, grace_period=50, reduction_factor=2  # 10 training iterations
    )

    tuner = tune.Tuner(
        trainer,
        tune_config=tune.TuneConfig(
            metric="eval-R2",
            mode="max",
            scheduler=scheduler,
            num_samples=100,
            # resources_per_trial={"cpu": 3, "gpu": 0.25},
        ),
        run_config=RunConfig(local_dir=os.path.join('/mnt/Incremental-Backup/others',experiment),
                             name=f'cluster_{c}'
                             ),
        param_space=search_space,
    )
    results = tuner.fit()





    best_bst = get_best_model_checkpoint(results)


    pred_test=best_bst.predict(xgb.DMatrix(submission_data))
    pred_test=scl_biomass.inverse_transform(pred_test.reshape(-1,1)).reshape(-1)
    cluster_predictions.append(pred_test)



pred_test=np.stack(cluster_predictions).mean(axis=0)+65.0
preds = pd.DataFrame({'ID':Submission["ID"],'Target':pred_test})
#%%
preds.to_csv(os.path.join(result_dir,f'Submission-embeddings-{embedding_experiment}.csv'), index=False)

