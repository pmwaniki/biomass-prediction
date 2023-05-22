import os
from pathlib import Path


home_dir=os.path.expanduser("~")
data_dir=os.path.join(home_dir,'data/biomass/data')
result_dir=os.path.join(home_dir,'data/biomass/results')
log_dir=os.path.join(home_dir,'experiments/biomass/logs')

if not os.path.exists(data_dir):
    # os.mkdir(data_dir)
    Path(data_dir).mkdir(parents=True,exist_ok=True)

if not os.path.exists(result_dir):
    # os.mkdir(result_dir)
    Path(result_dir).mkdir(parents=True,exist_ok=True)

if not os.path.exists(log_dir):
    # os.mkdir(log_dir)
    Path(log_dir).mkdir(parents=True,exist_ok=True)