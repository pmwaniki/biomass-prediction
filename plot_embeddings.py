
import os


import joblib
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as cl

from sklearn import manifold
from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt





from settings import data_dir, result_dir, log_dir
from loaders import H5Dataset



embedding_experiment="VAE-tuned"
experiment="Xgboost-embeddings"

train_h5path=os.path.join(data_dir,"train_combined.h5")
submission_h5path=os.path.join(data_dir,"submission.h5")

Train=pd.read_csv(os.path.join(data_dir,"Train.csv"))
Submission=pd.read_csv(os.path.join(data_dir,"Submission.csv"))
embeddings=joblib.load(os.path.join(result_dir,f"Embeddings-{embedding_experiment}.joblib"))

# Train_embeddings=np.concatenate([train for train,sub in embeddings],axis=1)
# Submission_embeddings=np.concatenate([sub for train,sub in embeddings],axis=1)

cluster=0
X_embeddings=np.concatenate([embeddings[cluster][0],embeddings[cluster][1]],axis=0)
biomass=np.concatenate([Train['biomass'],np.array([np.nan]*len(Submission))],axis=0)
source = np.concatenate([Train['cluster'],np.array(['6']*len(Submission))],axis=0)


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,learning_rate=100)
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X_embeddings))



def plot_embedding2(X,y=None, title=None,cmap=cm.hot,ax=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    if y.dtype != "float":
        levels = {l:cm.Set1(i) for i,l in enumerate(np.unique(y))}
    else:
        X = X[~np.isnan(y), :]
        y = y[~np.isnan(y)]

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1],
                    color=cmap((y[i]-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))) if y.dtype=="float" else levels[y[i]],
                    # color=cm.hot(y[i]),
                    alpha=0.5,)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax.set_xticks([]), ax.set_yticks([])
    # # plt.legend()
    if title is not None:
        ax.set_title(title)



sample=np.random.choice(len(X_embeddings),size=1000,replace=False)

_,ax = plt.subplots(1,1,figsize=(12,12))
plot_embedding2(X_tsne[sample,:],biomass[sample],ax=ax)
plt.show()

_,ax = plt.subplots(1,1,figsize=(12,12))
plot_embedding2(X_tsne[sample,:],source[sample],ax=ax)
plt.legend()
plt.show()



