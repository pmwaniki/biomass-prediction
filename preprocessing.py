import os
import h5py
import joblib
import numpy as np
import pandas as pd

from settings import data_dir, result_dir
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans




trainset = h5py.File(os.path.join(data_dir,"09072022_1154_train.h5"), "r")
validateset = h5py.File(os.path.join(data_dir,"09072022_1154_val.h5"), "r")
testset = h5py.File(os.path.join(data_dir,"09072022_1154_test.h5"), "r")
submissionset = h5py.File(os.path.join(data_dir,"images_test.h5"), "r")
submissionlat = h5py.File(os.path.join(data_dir,'TestFiles',"lat_test.h5"), "r")
submissionlon = h5py.File(os.path.join(data_dir,'TestFiles',"lon_test.h5"), "r")
submissioncloud = h5py.File(os.path.join(data_dir,'TestFiles',"cloud_test.h5"), "r")

train_h5=h5py.File(os.path.join(data_dir,"train_combined.h5"), "w")
submission_h5=h5py.File(os.path.join(data_dir,"submission.h5"), "w")
full_h5=h5py.File(os.path.join(data_dir,"full.h5"), "w")



# train
train_images = np.array(trainset['images'],dtype=np.float64)
train_images = train_images.transpose(0,3,1,2)
train_biomasses = np.array(trainset['agbd'],dtype=np.float64)
train_lon = np.array(trainset['lon'])[:,8,8,0]
train_lat = np.array(trainset['lat'])[:,8,8,0]
train_cloud=np.array(trainset['cloud']).mean(axis=(1,2,3))
train_data=pd.DataFrame({'lon':train_lon,'lat':train_lat,'biomass':train_biomasses,'cloud':train_cloud,'source':'train'})
# validate
validate_images = np.array(validateset['images'],dtype=np.float64)
validate_images = validate_images.transpose(0,3,1,2)
validate_biomasses = np.array(validateset['agbd'],dtype=np.float64)
validate_lon = np.array(validateset['lon'])[:,8,8,0]
validate_lat = np.array(validateset['lat'])[:,8,8,0]
validate_cloud=np.array(validateset['cloud']).mean(axis=(1,2,3))
validate_data=pd.DataFrame({'lon':validate_lon,'lat':validate_lat,'biomass':validate_biomasses,'cloud':validate_cloud,'source':'validation'})
# test
test_images = np.array(testset['images'],dtype=np.float32)
test_images = test_images.transpose(0,3,1,2)
test_biomasses = np.array(testset['agbd'],dtype=np.float64)
test_lon = np.array(testset['lon'])[:,8,8,0]
test_lat = np.array(testset['lat'])[:,8,8,0]
test_cloud=np.array(testset['cloud']).mean(axis=(1,2,3))
test_data=pd.DataFrame({'lon':test_lon,'lat':test_lat,'biomass':test_biomasses,'cloud':test_cloud,'source':'test'})

# test
submission_images = np.array(submissionset['images'],dtype=np.float32)
submission_images = submission_images.transpose(0,3,1,2)
submission_lon = np.array(submissionlon['lon'])[:,8,8,0]
submission_lat = np.array(submissionlat['lat'])[:,8,8,0]
submission_cloud=np.array(submissioncloud['cloud']).mean(axis=(1,2,3))
submission_data=pd.DataFrame({'lon':submission_lon,'lat':submission_lat,'cloud':submission_cloud,'source':'submission'})


Train=pd.concat([train_data,validate_data,test_data],axis=0)
Train_images=np.concatenate([train_images,validate_images,test_images],axis=0)
band_statistics={'mean':Train_images.mean(axis=(0,2,3)),'sd':Train_images.std(axis=(0,2,3))}
joblib.dump(band_statistics,os.path.join(data_dir,"band_statistics.joblib"))

Full=pd.concat([Train,submission_data],axis=0)
Full_images=np.concatenate([Train_images,submission_images,test_images],axis=0)

Full=Full.assign(lon_jittered=Full.lon+np.random.randn(Full.shape[0])*0.05,
                 lat_jittered=Full.lat+np.random.randn(Full.shape[0])*0.05)
fig,ax=plt.subplots(1,1,figsize=(10,10))
sns.scatterplot(Full.sample(n=Full.shape[0],replace=False),x="lon_jittered",y="lat_jittered",hue="source",ax=ax,alpha=0.5)
# ax.scatter(submission_data['lon'],submission_data['lat'],color="brown",label="Submission")
plt.savefig(os.path.join(result_dir,"Data source.png"))
plt.show()


clf=KMeans(n_clusters=5,random_state=123)
clf.fit(Train[['lon','lat']])
Train['cluster']=clf.predict(Train[['lon','lat']]).reshape(-1).astype("str")

fig,ax=plt.subplots(1,1,figsize=(10,10))
sns.scatterplot(Train,x="lon",y="lat",hue="cluster",ax=ax,alpha=0.5)
plt.savefig(os.path.join(result_dir,"Clusters.png"))
plt.show()


train_h5.create_dataset('images', data=Train_images)
train_h5.close()

full_h5.create_dataset('images', data=Full_images)
full_h5.close()

submission_h5.create_dataset('images', data=submission_images)
submission_h5.close()

Train=Train.reset_index(drop=True).rename_axis("S2_idx").reset_index()
Train.to_csv(os.path.join(data_dir,"Train.csv"),index=False)

sns.kdeplot(x='biomass',hue="cluster",data=Train,log_scale=True)
plt.show()

g = sns.FacetGrid(Train, col="cluster",col_wrap=3)
g.map(sns.kdeplot, "biomass")
plt.show()

Full=Full.reset_index(drop=True).rename_axis("S2_idx").reset_index()
Full.to_csv(os.path.join(data_dir,"Full.csv"),index=False)

ID_S2_pair = pd.read_csv(os.path.join(data_dir,'UniqueID-SentinelPair.csv'))

submission_data = submission_data.rename_axis('S2_idx').reset_index()
submission_data = submission_data.merge(ID_S2_pair, on='S2_idx')
submission_data.to_csv(os.path.join(data_dir,"Submission.csv"),index=False)


fig,ax=plt.subplots(1,1,figsize=(10,10))
sns.scatterplot(Train.assign(biomass=np.log(Train.biomass)).sample(n=Train.shape[0],replace=False),x="lon",y="lat",hue="biomass",ax=ax,alpha=0.5,)
# ax.scatter(submission_data['lon'],submission_data['lat'],color="brown",label="Submission")
plt.savefig(os.path.join(result_dir,"Biomass.png"))
plt.show()