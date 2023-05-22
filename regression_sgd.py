import copy

import h5py
import os


import joblib
import numpy as np
import pandas as pd
import torch
import kornia as K
from kornia.augmentation import AugmentationSequential

from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from modules import CNN
from settings import data_dir, result_dir, log_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ray
ray.init(address="auto")
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from loaders import H5Dataset

device="cuda"
experiment="Regression-SGD"



train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))










# def normalize(x):
#     return K.enhance.normalize_min_max(x)



configs = {
    'batch_size':tune.choice([128]),
    'lr':tune.loguniform(0.00001,0.1),
    'l2':tune.loguniform(0.000001,0.001),
    'dropout':tune.loguniform(0.01,0.7),
    'normalization_groups': tune.choice([0,1,2,6,12]),
    'gaussian_sd': tune.loguniform( 0.001,3.0),
    'hidden_dim': tune.choice([16,32,64]),
    'n_blocks': tune.choice([1,2,]),
    'clip_norm': tune.choice([0.1,1.0,2,10]),
    'kernel_size':tune.choice([1,3,5,7]),
    'blur_kernel_size':tune.choice([3,5]),
    'prop_noise':tune.choice([0.1,0.3,0.5,0.7,1.0])

}

config={i:v.sample() for i,v in configs.items()}

def get_model(config):
    model=CNN(dropout=config['dropout'],dim_out=1,hidden_dim=config['hidden_dim'],
                   normalization_groups=config['normalization_groups'],gaussian_sd=config['gaussian_sd'],
                   n_blocks=config['n_blocks'],kernel_size=config['kernel_size'])
    return model

def get_optimizer(config,model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'],)
    return optimizer

def get_train_loader(config,train_data):
    train_dataset = H5Dataset(train_data,train_h5path,return_outcome=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'],num_workers=5)
    return train_loader

def get_val_loader(val_data):
    val_dataset = H5Dataset(val_data,train_h5path,return_outcome=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32,num_workers=5)
    return val_loader

# Data augmentation
def augmentation(config):
    aug_list = AugmentationSequential(

        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        K.augmentation.RandomGaussianBlur(kernel_size=(config['blur_kernel_size'],config['blur_kernel_size']), sigma=(0.01, 2.0), p=config['prop_noise']),
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

def train_fun(model,optimizer,criterion,train_loader,val_loader,scheduler=None,clip_norm=1.0,aug=None):
    model.train()
    train_loss=0
    for batch_x,batch_y in train_loader:
        batch_x,batch_y=batch_x.to(device,dtype=torch.float),batch_y.to(device,dtype=torch.float).unsqueeze(1)
        if aug is not None:
            batch_x=aug(batch_x)
        # batch_x=normalize(batch_x)
        # batch_x = batch_x - 0.5
        pred=model(batch_x)
        loss=criterion(pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),clip_norm)
        optimizer.step()
        train_loss += loss.item() / len(train_loader)

    model.eval()
    val_loss = 0
    pred_val = []
    obs_val = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device,dtype=torch.float).unsqueeze(1)
            # batch_x=normalize(batch_x)
            # batch_x=batch_x - 0.5
            pred = model(batch_x)

            loss = criterion(pred, batch_y)
            val_loss += loss.item() / len(val_loader)
            pred_val.append(pred.squeeze(1).cpu().numpy().reshape(-1))
            obs_val.append(batch_y.squeeze(1).cpu().numpy().reshape(-1))
    if scheduler: scheduler.step()
    pred_val = np.concatenate(pred_val)
    obs_val = np.concatenate(obs_val)
    r2 = clipped_r2(obs_val, pred_val)
    mse=clipped_mse(obs_val,pred_val)
    return train_loss,val_loss,r2,mse


cluster_predictions=[]
cluster_state_config=[]
cluster_train_pred=[]
clusters=range(5)
for c in clusters:
    train_data = Train.loc[Train['cluster'] != c, :].copy()
    val_data = Train.loc[Train['cluster'] == c, :].copy()
    scl_biomass = QuantileTransformer(output_distribution='normal', n_quantiles=100)
    train_data['biomass'] = scl_biomass.fit_transform(train_data['biomass'].values.reshape(-1, 1)).reshape(-1)
    val_data['biomass'] = scl_biomass.transform(val_data['biomass'].values.reshape(-1, 1)).reshape(-1)
    class Trainer(tune.Trainable):
        def setup(self, config):
            self.model=get_model(config).to(device)
            self.optimizer=get_optimizer(config,self.model)
            self.criterion=nn.MSELoss().to(device)
            self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.1)
            self.train_loader=get_train_loader(config,train_data)
            self.val_loader=get_val_loader(val_data)
            self.clip_norm=config['clip_norm']
            self.aug=augmentation(config)


        def step(self):
            train_loss,loss,r2,mse=train_fun(self.model,self.optimizer,self.criterion,self.train_loader,self.val_loader,
                                         self.scheduler,self.clip_norm,self.aug)
            return {'loss':loss,'r2':r2,'train_loss':train_loss,'mse':mse}

        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            model_state,optimizer_state=torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

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
        resume=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=False,
        raise_on_failed_trial=False,
        # max_failures=1
    )



    df = result.results_df
    metric='r2';mode="max"; scope='last'
    print(result.get_best_trial(metric,mode,scope=scope).last_result)
    # df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
    best_trial = result.get_best_trial(metric, mode, scope=scope)
    best_config=result.get_best_config(metric,mode,scope=scope)

    test_dataset=H5Dataset(Submission,submission_h5path,return_outcome=False)
    test_loader=DataLoader(test_dataset,shuffle=False,batch_size=4,num_workers=4)

    best_checkpoint=result.get_best_checkpoint(best_trial,metric,mode,return_path=True)
    model_state,_=torch.load(os.path.join(best_checkpoint,"model.pth"))
    m_state=copy.deepcopy(model_state)
    cluster_state_config.append((m_state,best_config))

    # best_trainer=Trainer(best_config)
    best_model=get_model(best_config)
    best_model.load_state_dict(model_state)
    best_model.to(device)
    # Test model accuracy

    best_model.eval()
    pred_test = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            # batch_x=normalize(batch_x)
            # batch_x = batch_x - 0.5
            pred = best_model(batch_x)
            pred_test.append(pred.squeeze(1).cpu().numpy())

    pred_test = np.concatenate(pred_test,axis=0)
    pred_test = scl_biomass.inverse_transform(pred_test.reshape(-1,1)).reshape(-1)
    cluster_predictions.append(pred_test)

    full_dataset=H5Dataset(Train,train_h5path,return_outcome=False)
    full_loader=DataLoader(full_dataset,batch_size=128,shuffle=False)
    pred_full = []
    with torch.no_grad():
        for batch_x in full_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            pred = best_model(batch_x)
            pred_full.append(pred.squeeze(1).cpu().numpy())

    pred_full = np.concatenate(pred_full, axis=0)
    pred_full = scl_biomass.inverse_transform(pred_full.reshape(-1, 1)).reshape(-1)
    cluster_train_pred.append(pred_full)

pred_test=np.stack(cluster_predictions).mean(axis=0)+65.0
preds = pd.DataFrame({'ID':Submission["ID"],'Target':pred_test})
#%%
preds.to_csv(os.path.join(result_dir,f'Submission-{experiment}.csv'), index=False)

torch.save(cluster_state_config,f=os.path.join(result_dir,f"{experiment}-weights.pth"))
pred_train=np.stack(cluster_train_pred).mean(axis=0)
joblib.dump(pred_train,os.path.join(result_dir,f'Train-predictions-{experiment}.joblib'))


