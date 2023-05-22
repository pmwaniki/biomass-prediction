import os

import h5py
import joblib
import torch
from torch.utils.data import Dataset

from settings import data_dir

band_statistics=joblib.load(os.path.join(data_dir,"band_statistics.joblib"))

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
            image = (image-band_statistics['mean'].reshape((12,1,1)))/band_statistics['sd'].reshape((12,1,1))
        image_tensor=torch.from_numpy(image.astype("float32"))
        if self.return_outcome:
            outcome=row['biomass']
            outcome_tensor=torch.tensor(outcome,dtype=torch.float)
            return image_tensor,outcome_tensor

        return image_tensor


class SampledH5Dataset(Dataset):
    def __init__(self,data,h5_path,return_outcome=False,len=1000):
        self.length=len
        self.data=data
        self.h5_path=h5_path
        # self.h5=h5py.File(h5_path, "r")
        self.return_outcome=return_outcome

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        rows=self.data.iloc[torch.multinomial(torch.ones(self.data.shape[0]),225,replacement=False),:]
        # pixels=[]
        sampled_row,sampled_column=torch.randint(15,size=(2,))
        ids=rows['S2_idx'].values
        ids=np.sort(ids)
        with h5py.File(self.h5_path, "r") as f:
            images=f['images'][ids]
        image=images[:,:,sampled_row.item(),sampled_column.item()]
        image=image.T.reshape(12,15,15)
        # with h5py.File(self.h5_path, "r") as f:
        #     for id in rows['S2_idx']:
        #         pixels.append(f['images'][id][:,sampled_row.item(),sampled_column.item()])
        # image=np.stack(pixels,axis=1).reshape(12,15,15)
        image_tensor=torch.from_numpy(image.astype("float32"))
        if self.return_outcome:
            outcome=rows['biomass'].median()
            outcome_tensor=torch.tensor(outcome,dtype=torch.float)
            return image_tensor,outcome_tensor

        return image_tensor