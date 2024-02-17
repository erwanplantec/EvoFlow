import jax
import numpy as np
from tqdm import tqdm
from typing import NamedTuple
from sklearn.cluster import HDBSCAN

from analysis.genomes import GenomeData

class SpeciesData(NamedTuple):
	St: list
	Sct: list
	nSt: list

def get_species(data, t):
    uP_pr = data.uPt_pr[t]
    ca = HDBSCAN()
    ca.fit(uP_pr)
    return ca.labels_

def compute_species_data(data: GenomeData):
	n_species = []
	St = []
	for t in tqdm(range(len(data.uPt))):
		lbls = get_species(data, t)
		St.append(lbls)
		n_species.append(lbls.max()+1)
	S_centroids = [jax.ops.segment_sum(up_pr, sp)/len(sp) 
				   for up_pr, sp in zip(data.uPt_pr, St)]
	return SpeciesData(St=St, Sct=S_centroids, nSt = n_species)