import functools

import h5py
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch
from matplotlib import cm
from pyro.distributions import constraints
from pyro.distributions.transforms import neural_autoregressive

from pyro.infer import Predictive, NUTS, MCMC, SVI, Trace_ELBO,JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import pyro.distributions as dist
from pyro import poutine
from pyro.nn import PyroSample, PyroModule
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
from torchvision.transforms import Normalize

from modules import CNN
from settings import data_dir,result_dir
from loaders import H5Dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cuda'

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))







def model_horseshoe(x,y=None):
    P = x.shape[1]

    # sample from horseshoe prior
    lambdas = pyro.sample("lambdas_regression", dist.HalfCauchy(torch.ones(P,device=device)).to_event(1))
    tau = pyro.sample("tau_regression", dist.HalfCauchy(torch.ones(1,device=device)))


    unscaled_betas = pyro.sample("unscaled_betas_regression", dist.Normal(torch.tensor(0.0,device=device), torch.ones(P,device=device)*0.1).to_event(1))
    scaled_betas = pyro.deterministic("betas_regression", tau * lambdas * unscaled_betas)
    bias = pyro.sample('bias_regression', dist.Normal(torch.tensor(0.,device=device),
                                           torch.tensor(10.0,device=device)))
    mean_function = pyro.deterministic('mean_regression', bias +x @ scaled_betas.reshape(-1,1))


    prec_obs = pyro.sample("prec_obs_regression", dist.Gamma(torch.tensor(3.0,device=device), torch.tensor(1.0,device=device)))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    with pyro.plate("data",x.shape[0]):
        pyro.sample("obs_regression", dist.Normal(mean_function, sigma_obs).to_event(1), obs=y)
    return mean_function


class Encoder(nn.Module):
    def __init__(self,dropout=0.01,hidden_dim=32,dim_input=12,dim_z=16,normalization_groups=0,gaussian_sd=0,n_blocks=1,expansion=2,kernel_size=3):
        super().__init__()
        self.input_layer=CNN(dropout=dropout,hidden_dim=hidden_dim,dim_input=dim_input,dim_out=hidden_dim,
                             normalization_groups=normalization_groups,gaussian_sd=gaussian_sd,n_blocks=n_blocks,
                             expansion=expansion,kernel_size=kernel_size)
        self.mid_layers=nn.Sequential(
            nn.Linear(hidden_dim+1,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU()
        )
        self.fc_mean=nn.Linear(hidden_dim,dim_z)
        self.fc_var=nn.Linear(hidden_dim,dim_z)
    def forward(self,x,y):
        x=self.input_layer(x)
        x=torch.cat([x,y],dim=-1)
        x=self.mid_layers(x)
        z_loc=self.fc_mean(x)
        z_scale=F.softplus(self.fc_var(x))+1e-5
        return z_loc,z_scale


class Decoder(nn.Module):

    def __init__(self,
                 dim_z : int = 32,
                 act_fn : nn.Module = nn.GELU):

        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim_z, 16*64),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(32, 12, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(12, 12, kernel_size=4, padding=1,stride=1),

        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x




class SSVAE(pyro.nn.PyroModule):
    def __init__(self,dim_z=4,dim_hidden=16,n_hidden=2,n_flow=2,n_hidden_flow=8):
        super().__init__()
        self.dim_z=dim_z
        self.encoder = Encoder(hidden_dim=dim_hidden, dim_z=dim_z, n_blocks=n_hidden)
        self.decoder = Decoder(dim_z=dim_z)
        self.encoder_y=CNN(hidden_dim=dim_hidden,dim_out=1,kernel_size=3)
        self.flows=[dist.transforms.affine_coupling(input_dim=dim_z) for _ in range(n_flow)]
        self.flows_modules=nn.ModuleList(self.flows)
        self.cuda()
    def model(self,x,y=None):
        pyro.module("vae",self)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.dim_z)))
        z_scale = x.new_ones(torch.Size((x.shape[0], self.dim_z)))

        prec_obs = pyro.sample("prec_obs",
                               dist.Gamma(torch.tensor(3.0, device=device), torch.tensor(1.0, device=device)))
        sigma_obs = 1.0 / torch.sqrt(prec_obs)

        with pyro.plate("data",x.shape[0]):
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            x_hat = self.decoder(z)
            pyro.sample("obs",dist.Normal(x_hat,sigma_obs).to_event(3),obs=x)


    def guide(self,x,y=None):
        pyro.module("vae",self)

        prec_obs_shape=pyro.param("prec_obs_shape",torch.tensor(3.0, device=device))
        prec_obs_scale = pyro.param("prec_obs_scale", torch.tensor(1.0, device=device))
        prec_obs = pyro.sample("prec_obs",
                               dist.Gamma(prec_obs_shape, prec_obs_scale))
        prec_obs_y_loc=pyro.param('prec_obs_y_loc',init_tensor=torch.tensor(1.0,device=device),constraint=constraints.positive)
        prec_obs_y=pyro.sample('prec_obs_y',dist.Delta(prec_obs_y_loc),infer={'is_auxiliary': True})
        with pyro.plate("data",x.shape[0]):
            y_ = self.encoder_y(x)
            y_hat = pyro.sample('y_hat',dist.Normal(y_, prec_obs_y).to_event(1))

            z_loc,z_scale=self.encoder(x,y_hat)
            pyro.sample("latent",dist.Normal(z_loc,z_scale).to_event(1))
    def model_supervised(self,x,y=None):
        pyro.module("vae",self)
        prec_obs_y = pyro.sample('prec_obs_y', dist.Uniform(torch.tensor(0.001,device=device),torch.tensor(10.0,device=device)))
        with pyro.plate('data',x.shape[0]):
            y_hat=self.encoder_y(x)
            with poutine.scale(scale=1.0):
                pyro.sample('y',dist.Normal(y_hat,prec_obs_y).to_event(1),obs=y)
    def guide_supervised(self,x,y=None):
        prec_obs_y_loc = pyro.param('prec_obs_y_loc', init_tensor=torch.tensor(1.0, device=device),
                                    constraint=constraints.positive)
        prec_obs_y = pyro.sample('prec_obs_y', dist.Delta(prec_obs_y_loc))

    def flow_guide(self,x,y):
        pyro.module("vae",self)
        [pyro.module(f'flows_{i}',self.flows[i]) for i in range(len(self.flows))]
        prec_obs_shape=pyro.param("prec_obs_shape",torch.tensor(3.0, device=device))
        prec_obs_scale = pyro.param("prec_obs_scale", torch.tensor(1.0, device=device))
        prec_obs = pyro.sample("prec_obs",
                               dist.Gamma(prec_obs_shape, prec_obs_scale))
        with pyro.plate("data",x.shape[0]):
            z_loc,z_scale=self.encoder(x)

            pyro.sample("latent",dist.TransformedDistribution(dist.Normal(z_loc, z_scale),self.flows))

    def reconstruct(self,x):
        with torch.no_grad():
            z_loc,z_scale=self.encoder(x)
            # z=dist.Normal(z_loc,z_scale).sample()
            x_hat=self.decoder(z_loc)
        return x_hat
    def classify(self,x):
        y=self.encoder_y(x)
        return y



train_data = Train.loc[Train['cluster'] != 3, :].copy()
val_data = Train.loc[Train['cluster'] == 3, :].copy()

scl_biomass=QuantileTransformer(output_distribution='normal')
train_data['biomass']=scl_biomass.fit_transform(train_data['biomass'].values.reshape(-1,1)).reshape(-1)
val_data['biomass']=scl_biomass.transform(val_data['biomass'].values.reshape(-1,1)).reshape(-1)




train_dataset = H5Dataset(train_data,train_h5path,return_outcome=True)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128,num_workers=5,drop_last=True)

val_dataset = H5Dataset(val_data,train_h5path,return_outcome=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128,num_workers=5)

optimizer=ClippedAdam({'lr':1e-4,'lrd':0.99})
optimizer_supervised=ClippedAdam({'lr':1e-3})

vae=SSVAE(dim_z=32,n_hidden=1,dim_hidden=32,n_flow=2)
elbo=Trace_ELBO()
svi=SVI(vae.model,vae.guide,optimizer,loss=elbo)
# guide_supervised=pyro.infer.autoguide.AutoDelta(vae.model_supervised)
svi_supervised=SVI(vae.model_supervised,vae.guide_supervised,optimizer_supervised,loss=elbo)
# svi_supervised=SVI(model_horseshoe,guide_supervised,optimizer,loss=elbo)
pyro.clear_param_store()

train_elbo = []
test_elbo = []
for epoch in range(250):
    epoch_loss=0.0

    for i,(x,y) in enumerate(train_loader):
        batch_x = x.cuda()
        if i % 2 == 0:
            batch_y = y.cuda().reshape(-1, 1)
            loss=svi_supervised.step(batch_x,batch_y)
        else:
            batch_y=None
            loss=svi.step(batch_x,batch_y)
        epoch_loss +=loss/len(train_loader)
    train_elbo.append(epoch_loss)

    if epoch % 1 == 0:
        # initialize loss accumulator
        test_loss = 0.0
        test_rmse=0
        # compute the loss over the entire test set
        with torch.no_grad():
            for x,y in val_loader:
                batch_x = x.cuda()
                batch_y = y.cuda().reshape(-1, 1)
                loss = svi_supervised.evaluate_loss(batch_x,batch_y)
                test_loss += loss/len(val_loader)
                y_hat=vae.encoder_y(batch_x)
                test_rmse+=torch.pow(y_hat-batch_y,2).sum().item()/y_hat.shape[0]/len(val_loader)

        print("Test elbo: ", test_loss, "RMSE :", test_rmse)
        test_elbo.append(test_loss)


plt.plot(train_elbo,label="train")
plt.plot(test_elbo,label='val')
plt.legend()
plt.show()


pred_val=[]
with torch.no_grad():
    for x, _ in val_dataset:
        batch_x = x.cuda().permute((1, 2, 0)).reshape((-1, 12))
        y_hat = vae.classify(batch_x).mean().item()
        pred_val.append(y_hat)
pred_val=np.array(pred_val)

fig2,ax2=plt.subplots(1)
ax2.scatter(pred_val,val_data['biomass'],)
# ax2.set_xlim((0,1750))
ax2.plot([0,np.max(pred_val)],[0,np.max(pred_val)],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
plt.savefig(os.path.join(result_dir,"biomass-metric-learning.png"))
plt.show()