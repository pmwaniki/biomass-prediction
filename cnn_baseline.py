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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
x_train=K.enhance.normalize_min_max(x_train)
y_train=torch.tensor(train_biomasses,device=device)

x_val=torch.tensor(validate_images,device=device)
x_val=K.enhance.normalize_min_max(x_val)
y_val=torch.tensor(validate_biomasses,device=device)

x_test=torch.tensor(test_images,device=device)
x_test=K.enhance.normalize_min_max(x_test)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# self=CNN()
# x=torch.randn((5,12,15,15))
def get_model(config):
    return CNN()

def get_optimizer(model,config):
    pass

train_dataset=TensorDataset(x_train,y_train)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=32)

