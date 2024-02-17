from models.simple import Config as SimpleConfig
from models.dissipative import Config as DissipativeConfigs
from models.food import Config as FoodConfig

import os
from typing import Optional
try:
	import _pickle as pickle #type:ignore
except:
	import pickle
import gzip
from tqdm import tqdm

SAVE_DIR = "../evoflow_saves"

def load_file(pth):
    with gzip.GzipFile(pth, "rb") as file:
        o = pickle.load(file)
    return o

def load_run_data(run_name: str, seed: Optional[int]=None, return_config: bool=True):
	pth = f"{SAVE_DIR}/{run_name}"
	if seed is None:
		seeds = list(range(5))
	else:
		seeds = [seed]
	
	data = []
	for s in seeds:
		seed_data = []
		seedpth = f"{pth}/{s}"
		fnames = os.listdir(seedpth)
		fnames = [fname for fname in sorted(fnames, key = lambda f: int(f.split(".")[0]))]
		for fname, _ in zip(fnames, tqdm(range(len(fnames)))):
			d = load_file(f"{seedpth}/{fname}")
			seed_data.append(d)
		data.append(seed_data)
	
	if seed is not None:
		data = data[0]
	
	if return_config:
		cfg_pth = f"{pth}/config.pickle"
		cfg = load_file(cfg_pth)
		return data, cfg
	return data
