import os
import pandas as pd
import numpy as np

from settings import result_dir

regression_sgd=pd.read_csv(os.path.join(result_dir,"Submission-Regression-SGD.csv")).rename(columns={'Target':"M1"}).set_index("ID")
bayesian_tuned=pd.read_csv(os.path.join(result_dir,"Submission-Bayesian-tuned.csv")).rename(columns={'Target':"M2"}).set_index("ID")
embeddings_regression_vae=pd.read_csv(os.path.join(result_dir,"Submission-embeddings-VAE-tuned.csv")).rename(columns={'Target':"M3"}).set_index("ID")
embeddings_regression_flow=pd.read_csv(os.path.join(result_dir,"Submission-embeddings-VAE-tuned-flow.csv")).rename(columns={'Target':"M4"}).set_index("ID")
jtt=pd.read_csv(os.path.join(result_dir,"Submission-JTT-Regression-SGD.csv")).rename(columns={'Target':"M4"}).set_index("ID")

pred=pd.concat([
    regression_sgd,
    bayesian_tuned,
    embeddings_regression_vae,
    embeddings_regression_flow,
    jtt,
],axis=1)
pred2=pred.apply(np.mean,axis=1).reset_index().rename(columns={0:"Target"})

pred2.to_csv(os.path.join(result_dir,f'Submission-stacking.csv'), index=False)