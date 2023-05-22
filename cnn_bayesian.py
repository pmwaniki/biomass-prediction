import h5py
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch

from pyro.infer import Predictive, NUTS, MCMC, SVI, Trace_ELBO

import pyro.distributions as dist
from pyro import poutine
from pyro.nn import PyroSample, PyroModule
from pyro.nn.module import to_pyro_module_

from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.transforms import Normalize
import kornia.augmentation as K
from modules import CNN
from loaders import H5Dataset
from settings import data_dir,result_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cuda'

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))







# self=CNN()
# x=torch.randn((5,12,15,15),device=device)


class CNN2(pyro.nn.PyroModule):
    def __init__(self,hidden_dim=8):
        super().__init__()
        self.cnn=CNN(hidden_dim=hidden_dim,normalization_groups=2,dim_out=2,n_blocks=1,kernel_size=5,
                     dropout=0.012,gaussian_sd=0.0024).to(device)
        # self.fc_mean=nn.Sequential(*[
        #     nn.Linear(16,16),
        #     nn.LeakyReLU(),
        #     nn.Linear(16,1)
        # ])
        # self.fc_var = nn.Sequential(*[
        #     nn.Linear(16, 16),
        #     nn.LeakyReLU(),
        #     nn.Linear(16, 1)
        # ])

    def forward(self,x,y=None):
        x = self.cnn(x)
        # x_mean=self.fc_mean(x)
        # x_var=F.softplus(self.fc_var(x))
        x_mean = x[:,0]
        x_var = F.softplus(x[:,1])
        mean = pyro.deterministic("mean",x_mean)
        # sigma_obs = pyro.sample("sigma", dist.Uniform(torch.tensor(0.,device=device), torch.tensor(100.,device=device)))
        # prec_obs = pyro.sample("prec_obs", dist.Gamma(torch.tensor(1.0,device=device), torch.tensor(0.5,device=device)))
        # sigma_obs = 1.0 / torch.sqrt(prec_obs)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, x_var), obs=y)
        return mean




# model=CNNBayesian(hidden_dim=32,var_weights=0.01,var_bias=0.1).to(device)
model=CNN2(hidden_dim=64).to(device)
full_bayes=False
if full_bayes:
    to_pyro_module_(model.cnn)
    for m in model.modules():
        if True:#isinstance(m,nn.Conv2d):#pass
            # m.weight=PyroSample(dist.Normal(torch.tensor(0.0,device=device), torch.tensor(0.1,device=device)).expand(m.weight.shape).to_event(m.weight.dim()))
            for name, value in list(m.named_parameters(recurse=False)):
                if name in ('weight','bias'):
                    setattr(m, name, PyroSample(prior=dist.Normal(torch.tensor(0.0,device=device), torch.tensor(0.1,device=device))
                                                .expand(value.shape)
                                                .to_event(value.dim())))



# model=BayesianRegression(var_weights=10.0,var_bias=1.0).to(device)

# poutine.trace(model).get_trace(x_test.type(torch.float)).stochastic_nodes
# poutine.trace(model).get_trace(x_test.type(torch.float)).param_nodes

aug_list = K.AugmentationSequential(

        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.01, 2.0), p=.3),
        data_keys=["input",],
        same_on_batch=False,
        random_apply=10,keepdim=True
    ).to(device)


# guide_ppg.append(pyro.infer.autoguide.AutoMultivariateNormal(pyro.poutine.block(model_ppg,expose=['select_prob'])))
pyro.clear_param_store()


guide=pyro.infer.autoguide.AutoDelta(model)
# guide=pyro.infer.autoguide.AutoNormal(model)
guide=guide.to(device)

adam = pyro.optim.ClippedAdam({"lr": 0.00016,'betas':(0.99,0.999),'clip_norm':10.0,'weight_decay':0.0006})
svi_ppg = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=5))

train_data = Train.loc[Train['cluster'] != 4, :].copy()
val_data = Train.loc[Train['cluster'] == 4, :].copy()

scl_biomass=QuantileTransformer(output_distribution='normal',n_quantiles=100)
train_data['biomass']=scl_biomass.fit_transform(train_data['biomass'].values.reshape(-1,1)).reshape(-1)
val_data['biomass']=scl_biomass.transform(val_data['biomass'].values.reshape(-1,1)).reshape(-1)


train_dataset = H5Dataset(train_data,train_h5path,return_outcome=True)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128,num_workers=5)

val_dataset = H5Dataset(val_data,train_h5path,return_outcome=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=512,num_workers=5)

loss_=[]
val_loss_=[]
r2=[]
for j in range(3000):
    # calculate the loss and take a gradient step
    loss=0
    for x1,y_ in train_loader:#pass
        x1,y_=x1.to(device),y_.to(device)
        x1=aug_list(x1)
        l = svi_ppg.step(x1, y_)
        loss+=l/x1.shape[0]
    loss_.append(loss)
    r2_=0
    with torch.no_grad():
        for x1, y_ in val_loader:  # pass
            x1, y_ = x1.to(device), y_.to(device)
            l = svi_ppg.evaluate_loss(x1, y_)
            loss += l / x1.shape[0]
            y_pred=model(x1).detach().cpu().numpy()
            r2_ += r2_score(y_.detach().cpu().numpy(),y_pred)/len(val_loader)
        val_loss_.append(loss)
        r2.append(r2_)
    # loss = svi_ppg.step(x_data_ppg, y_data.reshape(-1,1))
    if j % 10 == 0:
        print(f"[iteration {j:04d}] loss: {loss_[-1]:.2f} <<>> val_loss: {val_loss_[-1]:.2f} <<>> R2: {r2_:.2f}" )


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
ax1.plot(loss_[2:],label="train")
ax1.plot(val_loss_[2:],label="val")
ax1.legend()
ax2.plot(r2,label="R2")
ax2.legend()
plt.show()

n_samples=100
predictive = Predictive(model, guide=guide, num_samples=n_samples,
                        return_sites=( "obs",))


val_samples = [predictive(x[0].to(device)) for x in val_loader]
val_samples=np.concatenate([v['obs'].cpu().numpy() for v in val_samples],axis=1)
pred_val=val_samples.mean(axis=0).squeeze().reshape(-1)
# pred_val=scl_biomass.inverse_transform(pred_val.reshape(-1,1)).reshape(-1)

print("R2: ",r2_baseline:=r2_score(val_data['biomass'].values,pred_val))
print("RMSE: ",rmse_baseline:=mean_squared_error(val_data['biomass'].values,pred_val,squared=False))

fig2,ax2=plt.subplots(1)
ax2.scatter(pred_val,val_data['biomass'].values,)
# ax2.set_xlim((0,1750))
ax2.plot([pred_val.min(),pred_val.max()],[pred_val.min(),pred_val.max()],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
plt.show()


plot_samples=pd.DataFrame(val_samples.T, columns=[f"s{i}" for i in range(n_samples)])
plot_samples['predicted']=plot_samples[[f"s{i}" for i in range(n_samples)]].mean(axis=1)
plot_samples['variance']=plot_samples[[f"s{i}" for i in range(n_samples)]].var(axis=1)
plot_samples['observed']=val_data['biomass'].values
plot_samples['error']=np.abs(plot_samples['predicted']-plot_samples['observed'])
fig2,ax=plt.subplots(1,figsize=(10,10))
sns.scatterplot(plot_samples,x='predicted',y="observed",hue='variance',ax=ax)
ax.plot([pred_val.min(),pred_val.max()],[pred_val.min(),pred_val.max()],'r--')
plt.show()

random_indices=np.random.choice(val_samples.shape[1],size=30,replace=False)
fig2,ax2=plt.subplots(1)
for i in random_indices:
    sns.kdeplot(x=val_samples[:,i],ax=ax2)

plt.show()



train2_loader=DataLoader(train_dataset,shuffle=False,batch_size=512)
train2_samples = [predictive(x.to(device)) for x,_ in train2_loader]
train2_samples=np.concatenate([v['obs'].cpu().numpy() for v in train2_samples],axis=1)
pred_train=train2_samples.mean(axis=0).squeeze().reshape(-1)


print("R2 train: ",r2_train:=r2_score(train_data['biomass'],pred_train))
sampled_points=np.random.choice(len(pred_train),size=1000,replace=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(pred_train[sampled_points],train_data['biomass'].iloc[sampled_points],)
# ax2.set_xlim((0,1750))
ax2.plot([pred_train.min(),pred_train.max()],[pred_train.min(),pred_train.max()],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
plt.show()

submission_dataset = H5Dataset(Submission,submission_h5path,return_outcome=False)
submission_loader = DataLoader(submission_dataset, shuffle=False, batch_size=32,num_workers=1)

submission_samples = [predictive(x.to(device)) for x in submission_loader]
submission_samples=np.concatenate([v['obs'].cpu().numpy() for v in submission_samples],axis=1)
pred_submission=np.exp(submission_samples.mean(axis=0).squeeze().reshape(-1))
pred_submission=scl_biomass.inverse_transform(pred_submission.reshape(-1,1)).reshape(-1)

preds = pd.DataFrame({'ID':Submission["ID"],'Target':pred_submission})
preds['Target']=preds['Target']+20.0
#%%
preds.to_csv(os.path.join(result_dir,f'Submission-Bayesian-CNN.csv'), index=False)