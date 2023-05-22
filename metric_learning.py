import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from kornia.augmentation import AugmentationSequential
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors._base import _get_weights
from sklearn.utils import check_array
from torch.nn import TripletMarginWithDistanceLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import kornia as K
from settings import data_dir, result_dir
from settings import log_dir
import os

from modules import MLPModel


from functools import partial

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.spatial.distance import cosine,pdist,cdist

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm

import ray
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")
ray.init(address="auto")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining,HyperBandScheduler,AsyncHyperBandScheduler

import joblib


display=os.environ.get('DISPLAY',None) is not None



enc_representation_size=16
enc_distance="DotProduct" #LpDistance Dotproduct Cosine
distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
pretext="proximity"
experiment=f"Contrastive-{pretext}-{enc_distance}{enc_representation_size}"

weights_file=os.path.join(result_dir,f"Triplet_{experiment}.pt")



#
Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))
train_data=Train.loc[Train['cluster'] != 4,:].copy()
val_data=Train.loc[Train['cluster'] == 4,:].copy()

dist=cdist(train_data[['lon','lat']],train_data[['lon','lat']],metric="euclidean")
dist=dist + np.eye(dist.shape[0]) * 1e3
train_data['clossest']=train_data.iloc[np.argmin(dist,axis=0),:]['S2_idx'].values
del dist

dist=cdist(val_data[['lon','lat']],val_data[['lon','lat']],metric="euclidean")
dist=dist + np.eye(dist.shape[0]) * 1e3
val_data['clossest']=val_data.iloc[np.argmin(dist,axis=0),:]['S2_idx'].values
del dist

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

class H5Dataset(Dataset):
    def __init__(self,data,h5_path,return_outcome=False):
        self.data=data
        self.h5_path=h5_path
        # self.h5=h5py.File(h5_path, "r")
        self.return_outcome=return_outcome

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        row=self.data.iloc[idx]
        with h5py.File(self.h5_path, "r") as f:
            image=f['images'][row['S2_idx']]
        image_tensor=torch.from_numpy(image.astype("float32"))
        if self.return_outcome:
            outcome=row['biomass']
            outcome_tensor=torch.tensor(outcome,dtype=torch.float)
            return image_tensor,outcome_tensor

        return image_tensor

class Triplets(Dataset):
    def __init__(self,data,h5_path,return_outcome=False):
        self.data=data
        self.h5_path=h5_path
        self.return_outcome=return_outcome

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        anchor_row=self.data.iloc[idx]
        positive_row=self.data.loc[self.data['S2_idx']==anchor_row['clossest'],:].iloc[0]
        negative_row=self.data.loc[self.data.cluster==anchor_row['cluster'],:]
        negative_row=negative_row.loc[~negative_row['S2_idx'].isin([positive_row['S2_idx'],anchor_row['S2_idx']]),:]
        negative_row=negative_row.iloc[torch.randint(negative_row.shape[0],(1,))].iloc[0]
        with h5py.File(os.path.join(data_dir,self.h5_path), "r") as f:
            anchor=f['images'][anchor_row['S2_idx']]
            positive=f['images'][positive_row['S2_idx']]
            negative=f['images'][negative_row['S2_idx']]
        anchor = torch.tensor(anchor.astype("float32"))
        positive = torch.tensor(positive.astype("float32"))
        negative = torch.tensor(negative.astype("float32"))

        if self.return_outcome:
            anchor_outcome=anchor_row['biomass']
            positive_outcome = positive_row['biomass']
            negative_outcome = negative_row['biomass']
            outcome_tensor = torch.tensor(np.array([anchor_outcome,
                                                    positive_outcome,
                                                    negative_outcome]), dtype=torch.float)
            return anchor, positive, negative, outcome_tensor
        return anchor,positive,negative

# self=Triplets(train_data,train_h5path,True)

encoder_val_dataset = Triplets(val_data,h5_path=train_h5path)
encoder_val_loader = DataLoader(encoder_val_dataset, batch_size=16, shuffle=False, num_workers=5)
#
# train_dataset = TriagePairs(train_encoder, id_var="id", stft_fun=None,
#                             transforms=None,
#                             # aug_raw=aug_raw,normalize=True
#                             )
# encoder_train_loader = DataLoader(encoder_train_dataset, batch_size=enc_batch_size, shuffle=True, num_workers=50)
#
# encoder_test_dataset = TriagePairs(test_encoder, id_var="id", stft_fun=None, aug_raw=[],normalize=True)
# encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=16, shuffle=False, num_workers=5)
#
classifier_train_dataset=Triplets(train_data,h5_path=train_h5path)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=16,shuffle=False,num_workers=5)

# classifier_test_dataset=TriageDataset(test,normalize=True)
# classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=16,shuffle=False,num_workers=5)
#
configs = {
    'dropout':tune.choice([0.001,0.005,0.01,0.02,0.05]),
    'representation_size':tune.choice([8,]),
    'batch_size':tune.choice([32,]),
    'lr':tune.loguniform(0.000001,0.0001),
    'l2':tune.loguniform(0.000001,0.0001),
    'delta': tune.choice([0.1,0.2,0.3,0.5]),
    'gaussian_sd':tune.choice([0.0,0.001,0.005,0.01]),
    'hidden_dim':tune.choice([16,]),
    'n_blocks': tune.choice([1,2,3,4]),
    'beta_adam': tune.choice([0.5,0.7,0.9,0.99])

}

config={i:v.sample() for i,v in configs.items()}



def normalize(x):
    return x / torch.norm(x, dim=1).unsqueeze(1).add(1e-6)

def accuracy_fun(x1,x2,x3):
    # similarity=nn.CosineSimilarity(dim=1,eps=1e-6)
    # similarity = lambda  a,b: -torch.cdist(a, b, p=2)
    similarity = lambda a, b: torch.einsum('bn,bn->b', a, b)
    sim_positive=similarity(x1,x2)
    sim_negative=similarity(x1,x3)
    accuracy=torch.mean((sim_positive>sim_negative).type(torch.float32)).item()
    return accuracy

def get_loader(config):
    classifier_train_dataset = Triplets(train_data,h5_path=train_h5path)
    encoder_val_dataset = Triplets(val_data,h5_path=train_h5path)
    train_loader = DataLoader(classifier_train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=5,drop_last=True)
    val_loader = DataLoader(encoder_val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=5,drop_last=True)
    return train_loader,val_loader

def get_model(config):
    model = MLPModel( dim_out=config['representation_size'],hidden_dim=config['hidden_dim'],
                       dropout=config['dropout'],normalization_groups=1,
                      gaussian_sd=config['gaussian_sd'],n_blocks=config['n_blocks'])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['l2'],betas=(config['beta_adam'],0.999))

    return optimizer

def augmentation(config):
    aug_list = AugmentationSequential(

        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        K.augmentation.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0), p=0.5),
        data_keys=["input",],
        same_on_batch=False,
        random_apply=10,keepdim=True
    ).to(device)
    return aug_list

device = "cuda" if torch.cuda.is_available() else "cpu"




def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler=None,aug=None):
    train_loss = 0
    train_accuracy=0
    model.train()
    # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    for x1, x2, x3 in train_loader:
        x1 = x1.to(device, dtype=torch.float)
        if aug is not None: x1=aug(x1)
        x1_emb = model(x1)
        x2 = x2.to(device, dtype=torch.float)
        if aug is not None: x2=aug(x2)
        x2_emb = model(x2)
        x3 = x3.to(device, dtype=torch.float)
        if aug is not None: x3=aug(x3)
        x3_emb = model(x3)

        x1_emb = normalize(x1_emb)
        x2_emb = normalize(x2_emb)
        x3_emb = normalize(x3_emb)

        loss = criterion(x1_emb,x2_emb,x3_emb)
        # l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).mean()
        # loss += l1_reg
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        train_loss += loss.item() / len(train_loader)
        train_accuracy += accuracy_fun(x1_emb,x2_emb,x3_emb)/len(train_loader)
        # metrics=calculator.get_accuracy(xis.cpu().detach().numpy(),xjs.cpu().detach().numpy(),
        #                         np.arange(xis.shape[0]),np.arange(xis.shape[0]),
        #                         False)

    model.eval()
    val_loss = 0
    val_accuracy=0
    # xis_embeddings = []
    # xjs_embeddings = []
    with torch.no_grad():
        for x1, x2, x3 in val_loader:
            x1 = x1.to(device, dtype=torch.float)
            x1_emb = model(x1)
            x2 = x2.to(device, dtype=torch.float)
            x2_emb = model(x2)
            x3 = x3.to(device, dtype=torch.float)
            x3_emb = model(x3)

            x1_emb = normalize(x1_emb)
            x2_emb = normalize(x2_emb)
            x3_emb = normalize(x3_emb)

            loss = criterion(x1_emb, x2_emb, x3_emb)

            val_loss += loss.item() / len(val_loader)
            val_accuracy += accuracy_fun(x1_emb, x2_emb, x3_emb) / len(val_loader)

        if scheduler is not None: scheduler.step()

    return train_loss,val_loss,train_accuracy,val_accuracy


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
        # self.scheduler=StepLR(self.optimizer,step_size=500,gamma=.5)
        self.scheduler = None
        # self.criterion=nn.TripletMarginLoss(margin=1.0, p=2.0).to(device)
        def cosine_dist(x,y):
            return 1.0 - F.cosine_similarity(x, y)
        def dot_dist(x,y):
            return 1.0 - torch.einsum('bn,bn->b', x, y)
        self.criterion=nn.TripletMarginWithDistanceLoss(distance_function=dot_dist,
                                                        margin=config['delta'])
        self.train_loader,self.val_loader=get_loader(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aug=augmentation(None)

    def step(self):
        train_loss,loss,train_accuracy,val_accuracy=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader,self.scheduler,self.aug)
        return {'loss':loss,'train_accuracy':train_accuracy,'train_loss':train_loss,'val_accuracy':val_accuracy}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)





scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=1000,
        grace_period=100,
        reduction_factor=2)

# scheduler = PopulationBasedTraining(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         perturbation_interval=10,
#         hyperparam_mutations={
#             # distribution for resampling
#             "enc_lr": lambda: np.random.uniform(1e-1, 1e-5),
#             "enc_l2": lambda: np.random.uniform(1e-1, 1e-5),
#         })

# scheduler=AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         grace_period=5,
#         max_t=700)
# scheduler = HyperBandScheduler(metric="loss", mode="min")



reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "val_accuracy", "training_iteration"])
result = tune.run(
    Trainer,
    # metric='loss',
    # mode='min',
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 4, "gpu": 0.33},
    config=configs,
    local_dir=log_dir,
    num_samples=100,
    name=experiment,
    resume=False,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False,)

df = result.results_df
metric='val_accuracy';mode='max'
best_trial = result.get_best_trial(metric, mode, "last-5-avg")
best_config=result.get_best_config(metric,mode,scope="last-5-avg")

best_model=get_model(best_config)
# best_trial=result.get_best_trial('loss','min')
best_checkpoint=result.get_best_checkpoint(best_trial,metric=metric,mode=mode,return_path=True)
model_state,optimizer_state=torch.load(os.path.join(best_checkpoint,"model.pth"))
torch.save((best_config,model_state),weights_file)

best_model.load_state_dict(model_state)
best_model.to(device)
# Test model accuracy


test_dataset=H5Dataset(Submission,submission_h5path,return_outcome=False)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=32)
best_model.eval()
test_embeddings = []
with torch.no_grad():
    for x1 in test_loader:
        x1 = x1.to(device, dtype=torch.float)
        xis = best_model(x1)
        xis = nn.functional.normalize(xis, dim=1)
        test_embeddings.append(xis.cpu().detach().numpy())
test_embeddings = np.concatenate(test_embeddings)

val_dataset=H5Dataset(val_data,train_h5path,return_outcome=True)
val_loader=DataLoader(val_dataset,shuffle=False,batch_size=32)
best_model.eval()
val_embeddings = []
val_y=[]
with torch.no_grad():
    for x1,y in val_loader:
        x1 = x1.to(device, dtype=torch.float)
        xis = best_model(x1)
        xis = nn.functional.normalize(xis, dim=1)
        val_embeddings.append(xis.cpu().detach().numpy())
        val_y.append(y.numpy())
val_embeddings = np.concatenate(val_embeddings)
val_y=np.concatenate(val_y)

train_dataset=H5Dataset(train_data,train_h5path,return_outcome=True)
train_loader=DataLoader(train_dataset,shuffle=False,batch_size=32)
best_model.eval()
train_embeddings = []
train_y=[]
with torch.no_grad():
    for x1,y in train_loader:
        x1 = x1.to(device, dtype=torch.float)
        xis = best_model(x1)
        xis = nn.functional.normalize(xis, dim=1)
        train_embeddings.append(xis.cpu().detach().numpy())
        train_y.append(y.numpy())
train_embeddings = np.concatenate(train_embeddings)
train_y=np.concatenate(train_y)



joblib.dump((train_embeddings,val_embeddings,test_embeddings,train_y,val_y),
            os.path.join(result_dir,f"{experiment}.joblib"))


x_train_reg,x_val_reg,y_train_reg,y_val_reg=train_test_split(val_embeddings,val_y)

class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')
        neigh_dist, neigh_ind = self.kneighbors(X)
        weights = _get_weights(neigh_dist, self.weights)
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))
        ######## Begin modification
        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification
        if self._y.ndim == 1:
            y_pred = y_pred.ravel()
        return y_pred

clf=GridSearchCV(estimator=MedianKNNRegressor(),param_grid={'n_neighbors':[1,3,5,10,15]},
                 scoring="neg_root_mean_squared_error",n_jobs=-1,cv=5)
clf.fit(x_train_reg,y_train_reg)
print("Best rmse on validation: " , clf.best_score_)
pred_val=clf.predict(x_val_reg)

fig2,ax2=plt.subplots(1)
ax2.scatter(pred_val,y_val_reg,)
# ax2.set_xlim((0,1750))
ax2.plot([0,40],[0,40],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
ax2.set_yscale("log")
ax2.set_xscale("log")
plt.savefig(os.path.join(result_dir,"biomass-metric-learning.png"))
plt.show()


pred_test=clf.predict(test_embeddings)
ID_S2_pair = pd.read_csv(os.path.join(data_dir,'UniqueID-SentinelPair.csv'))

preds = pd.DataFrame({'Target':pred_test}).rename_axis('S2_idx').reset_index()
preds = ID_S2_pair.merge(preds, on='S2_idx').drop(columns=['S2_idx'])
#%%
preds.to_csv(os.path.join(result_dir,f'GIZ_Biomass_predictions-{experiment}.csv'), index=False)