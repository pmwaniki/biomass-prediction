import copy
import functools

import h5py
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch
from matplotlib import cm
from pyro.distributions.transforms import neural_autoregressive,spline_autoregressive,spline_coupling

from pyro.infer import Predictive, NUTS, MCMC, SVI, Trace_ELBO,JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import pyro.distributions as dist
from pyro import poutine
from pyro.nn import PyroSample, PyroModule
from sklearn import manifold, linear_model, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, KFold
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
import kornia.augmentation as K
import ray
ray.init(address="auto")
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


from modules import CNN
from settings import data_dir, result_dir, log_dir
from loaders import H5Dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cuda'
use_flow=True
experiment="VAE-tuned"
if use_flow: experiment = experiment + "-flow"

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))







def model_horseshoe(x,y=None):
    P = x.shape[1]

    # sample from horseshoe prior
    lambdas = pyro.sample("lambdas_regression", dist.HalfCauchy(torch.ones(P)))
    tau = pyro.sample("tau_regression", dist.HalfCauchy(torch.ones(1)))


    unscaled_betas = pyro.sample("unscaled_betas_regression", dist.Normal(0.0, torch.ones(P)))
    scaled_betas = pyro.deterministic("betas_regression", tau * lambdas * unscaled_betas)
    bias = pyro.sample('bias_regression', dist.Normal(torch.tensor(0.),
                                           torch.tensor(1.)))
    mean_function = pyro.deterministic('mean_regression', bias +x @ scaled_betas.reshape(-1,1))


    prec_obs = pyro.sample("prec_obs_regression", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    pyro.sample("obs_regression", dist.Normal(mean_function, sigma_obs), obs=y)
    return mean_function




class Encoder(nn.Module):
    def __init__(self,dropout=0.01,hidden_dim=32,dim_input=12,dim_z=16,normalization_groups=0,gaussian_sd=0,n_blocks=1,expansion=2,kernel_size=3):
        super().__init__()
        self.input_layer=CNN(dropout=dropout,hidden_dim=hidden_dim,dim_input=dim_input,dim_out=hidden_dim,
                             normalization_groups=normalization_groups,gaussian_sd=gaussian_sd,n_blocks=n_blocks,
                             expansion=expansion,kernel_size=kernel_size)
        self.fc_mean=nn.Linear(hidden_dim,dim_z)
        self.fc_var=nn.Linear(hidden_dim,dim_z)
    def forward(self,x):
        x=self.input_layer(x)
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



class VAE(pyro.nn.PyroModule):
    def __init__(self,dim_z=4,dim_hidden=16,n_hidden=1,n_flow=2,flow_fun=spline_coupling):
        super().__init__()
        self.dim_z=dim_z
        self.encoder=Encoder(hidden_dim=dim_hidden,dim_z=dim_z,n_blocks=n_hidden)
        self.decoder=Decoder(dim_z=dim_z)
        self.flows=[flow_fun(input_dim=dim_z) for _ in range(n_flow)]
        self.flows_modules=nn.ModuleList(self.flows)
        self.cuda()
    def model(self,x):
        pyro.module("decoder",self.decoder)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.dim_z)))
        z_scale = x.new_ones(torch.Size((x.shape[0], self.dim_z)))

        prec_obs = pyro.sample("prec_obs",
                               dist.Gamma(torch.tensor(3.0, device=device), torch.tensor(1.0, device=device)))
        sigma_obs = 1.0 / torch.sqrt(prec_obs)

        with pyro.plate("data",x.shape[0]):
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            x_hat = self.decoder(z)
            y=pyro.sample("obs",dist.Normal(x_hat,sigma_obs).to_event(3),obs=x)

    def guide(self,x):
        pyro.module("encoder",self.encoder)
        prec_obs_shape=pyro.param("prec_obs_shape",torch.tensor(3.0, device=device))
        prec_obs_scale = pyro.param("prec_obs_scale", torch.tensor(1.0, device=device))
        prec_obs = pyro.sample("prec_obs",
                               dist.Gamma(prec_obs_shape, prec_obs_scale))
        with pyro.plate("data",x.shape[0]):
            z_loc,z_scale=self.encoder(x)
            pyro.sample("latent",dist.Normal(z_loc,z_scale).to_event(1))

    def flow_guide(self,x):
        pyro.module("encoder",self.encoder)
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


configs = {
    'batch_size':tune.choice([128]),
    'lr':tune.loguniform(0.000001,0.001),
    'l2':tune.loguniform(0.000001,0.001),
    'dropout':tune.loguniform(0.01,0.7),
    'normalization_groups': tune.choice([0,1,2,6,12]),
    'gaussian_sd': tune.loguniform( 0.001,3.0),
    'dim_hidden': tune.choice([4,8,16,32,64]),
    'n_hidden': tune.choice([1,2,3]),
    'clip_norm': tune.choice([0.1,1.0,2,4.,6.]),
    'kernel_size':tune.choice([3,5,7]),
    'blur_kernel_size':tune.choice([3,5]),
    'prop_noise':tune.choice([0.01,0.05,0.1,0.3,0.5,0.7,1.0],),
    'dim_z':tune.choice([8,16,32]),
    'n_flow': tune.choice([1,2,3]),
    'num_particles':tune.choice([1,])

}

config={i:v.sample() for i,v in configs.items()}

def get_model(config):

    model=VAE(dim_z=config['dim_z'],dim_hidden=config['dim_hidden'],n_hidden=config['n_hidden'],
              n_flow=config['n_flow'],)

    return model

def get_optimizer(config):
    optimizer = pyro.optim.ClippedAdam({'lr':config['lr'],'weight_decay':config['l2']})
    return optimizer

def get_train_loader(config,train_data):
    train_dataset = H5Dataset(train_data,train_h5path,return_outcome=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'],num_workers=5)
    return train_loader

def get_val_loader(val_data):
    val_dataset = H5Dataset(val_data,train_h5path,return_outcome=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32,num_workers=5)
    return val_loader

def augmentation(config):
    aug_list = K.AugmentationSequential(

        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        K.RandomGaussianBlur(kernel_size=(config['blur_kernel_size'],config['blur_kernel_size']), sigma=(0.01, 2.0), p=config['prop_noise']),
        data_keys=["input",],
        same_on_batch=False,
        random_apply=10,keepdim=True
    ).to(device)
    return aug_list

def clipped_mse(y_true,y_pred):
    try:
        mse=mean_squared_error(y_true,y_pred,squared=False)
        return mse
    except:
        return np.nan
def clipped_r2(y_true,y_pred):
    try:
        r2=r2_score(y_true,y_pred)
        if r2<0:
            return 0
        return r2
    except:
        return 0

def regression_score(X,y):
    pipeline = Pipeline([
        ('threashold', VarianceThreshold()),
        ('scl', StandardScaler()),
        ('clf', Lasso())])
    clf = GridSearchCV(estimator=pipeline, param_grid={'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]},
                       scoring='r2', refit=True, n_jobs=-1, cv=5)
    clf.fit(X, y)
    score=cross_validate(clf.best_estimator_,X,y,cv=KFold(5),n_jobs=-1,scoring=("neg_root_mean_squared_error",'r2'))
    rmse=np.nanmean(score['test_neg_root_mean_squared_error'])
    r2=np.nanmean(score['test_r2'])
    return -rmse,r2

def train_fun(vae,optimizer,criterion,train_loader,val_loader,aug=None):
    if use_flow:
        svi = SVI(vae.model, vae.flow_guide, optimizer, loss=criterion)
    else:
        svi= SVI(vae.model, vae.guide, optimizer, loss=criterion)
    train_loss=0
    for batch_x,batch_y in train_loader:
        batch_x,batch_y=batch_x.to(device,dtype=torch.float),batch_y.to(device,dtype=torch.float)
        if aug is not None:
            batch_x=aug(batch_x)
        loss=svi.step(batch_x)
        train_loss += loss / len(train_loader)

    val_loss = 0
    val_embeddings = []
    obs_val = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            z_ = vae.encoder(batch_x)[0]
            # loss = criterion(pred, batch_y)
            loss = svi.evaluate_loss( batch_x)
            val_loss += loss/len(val_loader)
            val_embeddings.append(z_.cpu().numpy())
            obs_val.append(batch_y.numpy().reshape(-1))
    # if scheduler: scheduler.step()
    val_embeddings = np.concatenate(val_embeddings,axis=0)
    obs_val = np.concatenate(obs_val)
    mse,r2=regression_score(val_embeddings,obs_val)

    return train_loss,val_loss,r2,mse


cluster_embeddings=[]
cluster_state_config=[]
clusters=range(5)
for c in clusters:
    train_data = Train.loc[Train['cluster'] != c, :].copy()
    val_data = Train.loc[Train['cluster'] == c, :].copy()
    scl_biomass = QuantileTransformer(output_distribution='normal', n_quantiles=100)
    train_data['biomass'] = scl_biomass.fit_transform(train_data['biomass'].values.reshape(-1, 1)).reshape(-1)
    val_data['biomass'] = scl_biomass.transform(val_data['biomass'].values.reshape(-1, 1)).reshape(-1)
    class Trainer(tune.Trainable):
        def setup(self, config):
            self.vae=get_model(config).to(device)
            self.optimizer=get_optimizer(config)
            self.criterion=pyro.infer.Trace_ELBO(num_particles=config['num_particles'])
            # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.1)
            self.train_loader=get_train_loader(config,train_data)
            self.val_loader=get_val_loader(val_data)
            # self.clip_norm=config['clip_norm']
            self.aug=augmentation(config)


        def step(self):
            train_loss,loss,r2,mse=train_fun(self.vae,self.optimizer,self.criterion,self.train_loader,self.val_loader, self.aug)
            return {'loss':loss,'r2':r2,'train_loss':train_loss,'mse':mse}

        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save((self.vae.state_dict(),self.optimizer.get_state(),pyro.get_param_store().get_state()), checkpoint_path)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            model_state,optimizer_state,pyro_state=torch.load(checkpoint_path)
            self.vae.load_state_dict(model_state)
            self.optimizer.set_state(optimizer_state)
            pyro.get_param_store().set_state(pyro_state)

    epochs=150
    scheduler = ASHAScheduler(
            metric="r2",
            mode="max",
            max_t=epochs,
            grace_period=50,
            reduction_factor=4)



    reporter = CLIReporter( metric_columns=["loss","train_loss","r2","mse", "training_iteration"])
    # early_stopping=tune.stopper.EarlyStopping(metric='auc',top=10,mode='max',patience=10)
    result = tune.run(
        Trainer,
        # metric='loss',
        # mode='min',
        checkpoint_at_end=True,
        resources_per_trial={"cpu": 3, "gpu": 0.25},
        config=configs,
        local_dir=os.path.join(log_dir,experiment),
        num_samples=100,
        name=f'cluster_{c}',
        # stop=MaxIterStopper(),
        resume=False,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=False,
        raise_on_failed_trial=False,
        # max_failures=1
    )

    df = result.results_df
    metric = 'r2';
    mode = "max";
    scope = 'last'
    print(result.get_best_trial(metric, mode, scope=scope).last_result)
    # df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
    best_trial = result.get_best_trial(metric, mode, scope=scope)
    best_config = result.get_best_config(metric, mode, scope=scope)

    test_dataset = H5Dataset(Submission, submission_h5path, return_outcome=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)

    best_checkpoint = result.get_best_checkpoint(best_trial, metric, mode, return_path=True)
    model_state, _, _ = torch.load(os.path.join(best_checkpoint, "model.pth"))
    m_state = copy.deepcopy(model_state)
    cluster_state_config.append((m_state, best_config))

    # best_trainer=Trainer(best_config)
    best_model = get_model(best_config)
    best_model.load_state_dict(model_state)
    best_model.to(device)


    submission_embeddings = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            z_ = best_model.encoder(batch_x)[0]
            submission_embeddings.append(z_.cpu().numpy())

    submission_embeddings = np.concatenate(submission_embeddings, axis=0)

    train_dataset2=H5Dataset(Train,h5_path=train_h5path,return_outcome=False)
    train_loader2=DataLoader(train_dataset2,shuffle=False,batch_size=256)

    train_embeddings = []
    with torch.no_grad():
        for batch_x in train_loader2:
            batch_x = batch_x.to(device, dtype=torch.float)
            z_ = best_model.encoder(batch_x)[0]
            train_embeddings.append(z_.cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings, axis=0)

    cluster_embeddings.append((train_embeddings,submission_embeddings))

joblib.dump(cluster_embeddings,os.path.join(result_dir,f"Embeddings-{experiment}.joblib"))

# train_data = Train.loc[Train['cluster'] != 4, :].copy()
# val_data = Train.loc[Train['cluster'] == 4, :].copy()
# train_dataset = H5Dataset(train_data,train_h5path,return_outcome=True)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128,num_workers=5,drop_last=True)
#
# val_dataset = H5Dataset(val_data,train_h5path,return_outcome=True)
# val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128,num_workers=5)
#
# optimizer=Adam({'lr':1.0e-4,})
# vae=VAE(dim_z=32,n_hidden=2,dim_hidden=8 ,n_flow=10, flow_fun=dist.transforms.neural_autoregressive)
# elbo=Trace_ELBO()
# svi=SVI(vae.model,vae.guide,optimizer,loss=elbo)
# pyro.clear_param_store()
#
# train_elbo = []
# test_elbo = []
# for epoch in range(250):
#     epoch_loss=0.0
#
#     for x,_ in train_loader:
#         batch_x=x.cuda()
#         loss=svi.step(batch_x)
#         epoch_loss +=loss/len(train_loader)
#     train_elbo.append(epoch_loss)
#
#     if epoch % 1 == 0:
#         # initialize loss accumulator
#         test_loss = 0.0
#         # compute the loss over the entire test set
#         for x,_ in val_loader:
#             batch_x = x.cuda()
#             loss = svi.evaluate_loss(batch_x)
#             test_loss += loss/len(val_loader)
#         print("Test elbo: ", test_loss)
#         test_elbo.append(test_loss)
#
#
# plt.plot(train_elbo)
# plt.show()
#
# #get embeddings
# train_loader2=DataLoader(train_dataset,shuffle=False,num_workers=5,batch_size=128)
# train_embeddings=[]
# for x,y in train_loader2:
#     batch_x = x.cuda()
#     with torch.no_grad():
#         train_embeddings.append(vae.encoder(batch_x)[0].cpu().numpy())
#
# train_embeddings=np.concatenate(train_embeddings,axis=0)
# train_y=train_data['biomass'].values
#
#
# val_embeddings=[]
# for x,_ in val_loader:
#     batch_x = x.cuda()
#     with torch.no_grad():
#         val_embeddings.append(vae.encoder(batch_x)[0].cpu().numpy())
#
# val_embeddings=np.concatenate(val_embeddings,axis=0)
# val_y=val_data['biomass'].values
#
# submission_dataset = H5Dataset(Submission,submission_h5path,return_outcome=False)
# submission_loader=DataLoader(submission_dataset,shuffle=False,batch_size=10)
# submission_embeddings=[]
# for x in submission_loader:
#     batch_x = x.cuda()
#     with torch.no_grad():
#         submission_embeddings.append(vae.encoder(batch_x)[0].cpu().numpy())
#
# submission_embeddings=np.concatenate(submission_embeddings,axis=0)
#
#
#
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,learning_rate=100)
# X_tsne = tsne.fit_transform(StandardScaler().fit_transform(val_embeddings))
#
#
#
# def plot_embedding2(X,y=None, title=None,cmap=cm.hot,ax=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#     X=X[~np.isnan(y),:]
#     y=y[~np.isnan(y)]
#
#     if ax is None:
#         plt.figure()
#         ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         ax.scatter(X[i, 0], X[i, 1],
#                     color=cmap((y[i]-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))),
#                     # color=cm.hot(y[i]),
#                     alpha=0.5,)
#     ax.set_xlabel("Component 1")
#     ax.set_ylabel("Component 2")
#
#     ax.set_xticks([]), ax.set_yticks([])
#     # # plt.legend()
#     if title is not None:
#         ax.set_title(title)
#
# _,ax = plt.subplots(1,1,figsize=(12,12))
# plot_embedding2(X_tsne,val_y,ax=ax)
# plt.show()
#
#
# ########################################################################################################################
#
#
# ######################################################################################################################
# # x_train_reg,x_val_reg,y_train_reg,y_val_reg=train_test_split(val_embeddings,val_y,random_state=123)
# model_regression=model_horseshoe
# # params=poutine.trace(model_regression).get_trace(x_train_hr).stochastic_nodes
# # poutine.trace(model_hr).get_trace(x_train_hr).param_nodes
#
# pipe=Pipeline([
#     ('poly',PolynomialFeatures(degree=2,include_bias=False)),
#     ('scl',StandardScaler()),
#
# ])
#
# regression_kernel = NUTS(model_regression, full_mass=True )
# mcmc=MCMC(regression_kernel,num_samples=500,warmup_steps=200,num_chains=1)
# mcmc.run(torch.tensor(pipe.fit_transform(train_embeddings),dtype=torch.float),torch.tensor(train_y,dtype=torch.float).log().reshape(-1,1))
# posterior_samples=mcmc.get_samples().copy()
# predictive_hr=Predictive(model_regression,posterior_samples)
# samples=predictive_hr(torch.tensor(pipe.transform(val_embeddings),dtype=torch.float),None)
# # joblib.dump(posterior_samples,os.path.join(result_dir,f"RegressionBayesian_vae.joblib"))
#
# pred_val=samples['mean_regression'].mean(dim=0).reshape(-1).exp().cpu().numpy()
#
#
#
# pipeline=Pipeline([
#     # ('poly',PolynomialFeatures(include_bias=False)),
#     ('scl',StandardScaler(),),
#     ('clf',linear_model.Lasso(max_iter=100000))
# ])
# param_grid={}
# clf=GridSearchCV(estimator=pipeline,param_grid={'clf__alpha':[0.00001,0.0001,0.001,0.01,0.1]},
#                  scoring=["neg_root_mean_squared_error",'r2'],refit='r2',n_jobs=-1,cv=5)
# clf.fit(train_embeddings,train_y)
# print("Best rmse on validation: " , clf.best_score_)
# pred_val=clf.predict(val_embeddings)
#
# print("Val rmse",metrics.mean_squared_error(val_y,pred_val,squared=False))
# print("Val R2",metrics.r2_score(val_y,pred_val))
#
#
# fig2,ax2=plt.subplots(1)
# ax2.scatter(pred_val,val_y,)
# # ax2.set_xlim((0,1750))
# ax2.plot([0,np.max(pred_val)],[0,np.max(pred_val)],'r--')
# ax2.set_xlabel("Predicted")
# ax2.set_ylabel("Observed")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
# plt.savefig(os.path.join(result_dir,"biomass-metric-learning.png"))
# plt.show()
#
#
# pred_test=clf.predict(submission_embeddings)+65.0
# preds = pd.DataFrame({'ID':Submission["ID"],'Target':pred_test})
# #%%
# preds.to_csv(os.path.join(result_dir,f'Vanilla_vae-submission.csv'), index=False)











