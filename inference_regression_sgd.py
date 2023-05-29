import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader

from loaders import H5Dataset
from modules import CNN
from settings import result_dir,data_dir

device='cuda'
submission_h5path=os.path.join(data_dir,"submission.h5")
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))

"""
The train, validation and test data are combined into a single file. K-Mean clustering is used to create five
clusters using the long and latitude of the satellite images
"""
Train=pd.read_csv(os.path.join(data_dir,"Train.csv")) #


"""
file containing configuration and weights of best model. There are five sets of weights and configurations. One for each
cluster.
"""
#
weights_file=os.path.join(result_dir,"Regression-SGD-weights.pth")
config_weights=torch.load(weights_file)


test_dataset = H5Dataset(Submission, submission_h5path, return_outcome=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=50, num_workers=4)

def get_model(config):
    model=CNN(dropout=config['dropout'],dim_out=1,hidden_dim=config['hidden_dim'],
                   normalization_groups=config['normalization_groups'],gaussian_sd=config['gaussian_sd'],
                   n_blocks=config['n_blocks'],kernel_size=config['kernel_size'])
    return model

cluster_predictions=[]
for c in range(5): #5 predictions are made: one for each cluster
    train_data = Train.loc[Train['cluster'] != c, :].copy()
    scl_biomass = QuantileTransformer(output_distribution='normal', n_quantiles=100,random_state=69)
    train_data['biomass'] = scl_biomass.fit_transform(train_data['biomass'].values.reshape(-1, 1)).reshape(-1)
    model_state, best_config = config_weights[c]
    best_model = get_model(best_config)
    best_model.load_state_dict(model_state)
    best_model.to(device)
    # Test model accuracy

    best_model.eval()
    pred_test = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            pred = best_model(batch_x)
            pred_test.append(pred.squeeze(1).cpu().numpy())

    pred_test = np.concatenate(pred_test, axis=0)
    pred_test = scl_biomass.inverse_transform(pred_test.reshape(-1, 1)).reshape(-1)
    cluster_predictions.append(pred_test)

pred_test=np.stack(cluster_predictions).mean(axis=0)+65.0
preds = pd.DataFrame({'ID':Submission["ID"],'Target':pred_test})