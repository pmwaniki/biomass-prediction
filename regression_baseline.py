import h5py
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch
import kornia as K

from pyro.infer import  Predictive, NUTS, MCMC

import pyro.distributions as dist
from pyro import poutine

from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from torch import nn

from settings import data_dir,result_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cpu'
num_chains=1


trainset = h5py.File(os.path.join(data_dir,"09072022_1154_train.h5"), "r")
validateset = h5py.File(os.path.join(data_dir,"09072022_1154_val.h5"), "r")
testset = h5py.File(os.path.join(data_dir,"images_test.h5"), "r")


# train
train_images = np.array(trainset['images'],dtype=np.float64)
train_images = train_images.transpose(0,3,1,2)
train_biomasses = np.array(trainset['agbd'],dtype=np.float64)

# validate
validate_images = np.array(validateset['images'],dtype=np.float64)
validate_images = validate_images.transpose(0,3,1,2)
validate_biomasses = np.array(validateset['agbd'],dtype=np.float64)

# test
test_images = np.array(testset['images'],dtype=np.float32)
test_images = test_images.transpose(0,3,1,2)

x_train=torch.tensor(train_images,device=device)
# x_train=K.enhance.normalize_min_max(x_train)
y_train=torch.tensor(train_biomasses,device=device)

x_val=torch.tensor(validate_images,device=device)
# x_val=K.enhance.normalize_min_max(x_val)
y_val=torch.tensor(validate_biomasses,device=device)

x_test=torch.tensor(test_images,device=device)
# x_test=K.enhance.normalize_min_max(x_test)

def model_regression(x,y=None):
    P = (12,1,1)

    betas=pyro.sample("betas",dist.Normal(torch.tensor(0.0,device=device),
                                          torch.ones(P,device=device)))
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(1., device=device)))
    mean_function = pyro.deterministic('mean', torch.mean(bias + x * betas,dim=[1,2,3]))


    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    pyro.sample("obs", dist.Normal(mean_function, sigma_obs), obs=y)
    return mean_function


def model_horseshoe(x,y=None):
    P = (12,1,1)

    # sample from horseshoe prior
    lambdas = pyro.sample("lambdas", dist.HalfCauchy(torch.ones(P)))
    tau = pyro.sample("tau", dist.HalfCauchy(torch.ones(1)))


    unscaled_betas = pyro.sample("unscaled_betas", dist.Normal(0.0, torch.ones(P)))
    scaled_betas = pyro.deterministic("betas", tau * lambdas * unscaled_betas)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(10., device=device)))
    mean_function = pyro.deterministic('mean', torch.mean(bias + x * scaled_betas,dim=[1,2,3]))


    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    pyro.sample("obs", dist.Normal(mean_function, sigma_obs), obs=y)
    return mean_function
#******************************************************************************************************************

model_baseline=model_regression
params=poutine.trace(model_baseline).get_trace(x_train).stochastic_nodes
poutine.trace(model_baseline).get_trace(x_train).param_nodes

baseline_kernel = NUTS(model_baseline )
mcmc=MCMC(baseline_kernel,num_samples=200,warmup_steps=100,num_chains=num_chains)
mcmc.run(x_train,y_train.log())
posterior_samples_baseline=mcmc.get_samples().copy()
predictive_baseline=Predictive(model_baseline,posterior_samples_baseline)
test_samples=predictive_baseline(x_test,None)
y_test=test_samples['mean'].mean(axis=0).exp().numpy()

val_samples=predictive_baseline(x_val,None)
pred_val=val_samples['mean'].mean(axis=0).exp().numpy()

joblib.dump(posterior_samples_baseline,os.path.join(result_dir,f"RegressionBayesian-parameter samples.joblib"))




print("R2: ",r2_baseline:=r2_score(y_val,pred_val))
rmse_baseline=mean_squared_error(y_val,pred_val,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(pred_val,y_val,)
ax2.set_xlim((0,200))
# ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
plt.show()

ID_S2_pair = pd.read_csv(os.path.join(data_dir,'UniqueID-SentinelPair.csv'))

preds = pd.DataFrame({'Target':y_test}).rename_axis('S2_idx').reset_index()
preds = ID_S2_pair.merge(preds, on='S2_idx').drop(columns=['S2_idx'])
#%%
preds.to_csv(os.path.join(result_dir,f'GIZ_Biomass_predictions-baseline.csv'), index=False)



