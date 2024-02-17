import jax
import numpy as np
from tqdm import tqdm
from typing import NamedTuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GenomeData(NamedTuple):
    uPt: list # unique genomes at each time
    fuPt: np.ndarray # concatenated uniaue genomes
    fT: np.ndarray # timetesps matching concatenated genomes
    uPit: list # unique genome ids at each time
    cPt: list # cell counts for each genome at each time
    aPt: list # total matter for each genome at each time
    iPt: list # id of genome for each cell at each time
    n_uP: int # nb of different genomes
    At: list # flattened map of matter at each step
    uP: np.ndarray # all unique genomes
    T: np.ndarray
    # --- PCA ---
    uPt_pr: list # same than uPt but projected through PCA
    pca: PCA


def compute_genome_data(data: list):

	N = len(data) #number of data points
	T = np.arange(len(data))*100

	uPt = [d["uP"] for d in data]
	cPt = [d["cP"] for d in data]
	iPt = [d["iP"] for d in data]

	fuPt = np.concatenate(uPt)
	fT = np.concatenate([np.ones(u.shape[:1])*i for i, u in enumerate(uPt)]) * 100

	print("Computing unique genomes set")
	uP, Pi = np.unique(fuPt, axis=0, return_inverse=True)
	n_uP = uP.shape[0] 
	print(f"Found {n_uP} unique genomes")
	uPit = []
	i = 0
	for d in data:
		n = d["uP"].shape[0]
		uPit.append(Pi[i:i+n])
		i += n
	At = [d["A"].reshape((-1,)) for d in data]
	aPt = [jax.ops.segment_sum(a, ipt) for a, ipt in zip(At, iPt)]

	print("Computing pca projection")
	pca = PCA(n_components=5)
	pca.fit(uP)
	print("		explained variance: ", pca.explained_variance_ratio_)
	uPt_pr = [pca.transform(upt) for upt in uPt]

	return GenomeData(uPt=uPt, 
					  cPt=cPt, 
					  iPt=iPt, 
					  aPt=aPt, 
					  At=At, 
					  n_uP=n_uP, 
					  uP=uP,
                   	  uPit=uPit, 
                   	  fuPt=fuPt, 
                   	  fT=fT,
                   	  pca=pca,
                   	  uPt_pr=uPt_pr,
                   	  T=T)


def plot_genomes_projection(data: GenomeData, d=2, cmap="rainbow"):

	fig = plt.figure(figsize=(16,16))
	fuPt_pr = np.concatenate(data.uPt_pr)
	if d==3:
		ax = fig.add_subplot(projection='3d')
		ax.scatter(*fuPt_pr[:, :3].T, c=data.fT, cmap=cmap) #type:ignore
	else:
		ax = fig.add_subplot()
		ax.scatter(*fuPt_pr[:, :2].T, c=data.fT, cmap=cmap) #type:ignore

	plt.show()

def plot_genomes_projection_in_time(data: GenomeData, d=2, cmap="rainbow"):
	fig = plt.figure(figsize=(16,16))
	fuPt_pr = np.concatenate(data.uPt_pr)
	if d==2:
		ax = fig.add_subplot()
		ax.scatter(fuPt_pr[:,0], data.T, c=fuPt_pr[:,1], cmap=cmap)#type:ignore
	elif d==3:
		ax = fig.add_subplot(projection='3d')
		ax.scatter(fuPt_pr[:,0], fuPt_pr[:,1], data.T, c=fuPt_pr[:,2], cmap=cmap)#type:ignore
	plt.show()





