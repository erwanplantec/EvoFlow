import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Optional
import matplotlib.pyplot as plt

from analysis.genomes import GenomeData


class EAData(NamedTuple):
	A_p: np.ndarray
	A_c: np.ndarray
	A_n: np.ndarray
	a_p: Optional[np.ndarray]=None
	a_c: Optional[np.ndarray]=None
	a_n: Optional[np.ndarray]=None
	delta_p: Optional[np.ndarray]=None
	delta_c: Optional[np.ndarray]=None
	delta_n: Optional[np.ndarray]=None


def compute_ea_data(data: GenomeData, return_a: bool=False, return_deltas: bool=False)->EAData:
	n_species = data.n_uP
	ts = np.arange(len(data.uPt))

	A_p = np.zeros((len(ts),))
	A_c = np.zeros((len(ts),))
	A_n = np.zeros((len(ts),))

	if return_a:
		activities_p = np.zeros((len(ts), n_species), dtype=int)
		activities_c = np.zeros((len(ts), n_species), dtype=int)
		activities_n = np.zeros((len(ts), n_species), dtype=int)
	else:
		activities_p = None
		activities_c = None
		activities_n = None

	if return_deltas:
		deltas_p = np.zeros((len(ts), n_species), dtype=int)
		deltas_c = np.zeros((len(ts), n_species), dtype=int)
		deltas_n = np.zeros((len(ts), n_species), dtype=int)
	else:
		deltas_p = None
		deltas_c = None
		deltas_n = None

	a_p = np.zeros(n_species, dtype=int)
	a_c = np.zeros(n_species, dtype=int)
	a_n = np.zeros(n_species)

	props = np.zeros((n_species, ))
    
	for t in tqdm(range(len(data.uPt))):
		ps = data.uPit[t]
		cs = data.aPt[t]
		C = cs.sum()
		delta_c = np.zeros_like(a_c)
		delta_p = np.zeros_like(a_p)
		delta_n = np.zeros_like(a_n)
		exist = np.zeros_like(a_p)

		exist[ps] = 1
		delta_c[ps] = cs
		# --- Existence based ---
		delta_p[ps] = 1
		# --- Non neutral ---
		e = props[ps]
		pr = cs / C
		props[ps] = pr
		delta_n[ps] = np.where(pr>e, C*(pr-e)**2, 0.)
		    
		a_p = (a_p + delta_p) * exist
		a_c = (a_c + delta_c) * exist
		a_n = (a_n + delta_n) * exist

		A_p[t] = a_p.sum()
		A_c[t] = a_c.sum()
		A_n[t] = a_n.sum()

		if return_a:
			activities_p[t] = a_p #type:ignore
			activities_c[t] = a_c #type:ignore
			activities_n[t] = a_n #type:ignore
		if return_deltas:
			deltas_p[t] = delta_p #type:ignore
			deltas_c[t] = delta_c #type:ignore
			deltas_n[t] = delta_n #type:ignore

	return EAData(A_p=A_p, A_c=A_c, A_n=A_n, 
				  a_n=activities_n, a_p=activities_p, a_c=activities_c, 
				  delta_n=deltas_n, delta_p=deltas_p, delta_c=deltas_c)




def plot_ea_data(data: EAData, plot_A: bool=True, plot_a: bool=True, 
				 plot_d: bool=False, save_dir: Optional[str]=None):
	
	if plot_A:
		_, ax = plt.subplots(3, figsize=(16, 8), sharex=True)
		ax[0].plot(data.a_p.sum(-1))
		ax[0].set_title("AP")
		ax[1].plot(data.a_c.sum(-1))
		ax[1].set_title("AC")
		ax[2].plot(data.a_n.sum(-1))
		ax[2].set_title("AN")
		if save_dir is not None:
			plt.savefig(save_dir+"x=t_y=Ax.png")
		plt.show()
	
	if plot_a:
		_, ax = plt.subplots(3, figsize=(16, 8), sharex=True)
		for i , A in enumerate([data.a_p, data.a_c, data.a_n]):
			A_ = np.ones_like(A, dtype=float) * A
			mask = (A==0) & (np.roll(A, -1, axis=0)==0)
			A_[mask] = np.nan
			ax[i].plot(A_, linewidth=.3)
		ax[0].set_ylabel("a_P")
		ax[1].set_ylabel("a_C")
		ax[2].set_ylabel("a_N")
		if save_dir is not None:
			plt.savefig(save_dir+"x=t_y=aX.png")
		plt.show()

	if plot_d:
		assert data.delta_c is not None
		_, ax = plt.subplots(3, figsize=(16, 8), sharex=True)
		for i , A in enumerate([data.delta_p, data.delta_c, data.delta_n]):
			A_ = np.ones_like(A, dtype=float) * A
			mask = (A==0) & (np.roll(A, -1, axis=0)==0) #type:ignore
			A_[mask] = np.nan
			ax[i].plot(A_, linewidth=.3)
		ax[0].set_ylabel("dP")
		ax[1].set_ylabel("dC")
		ax[2].set_ylabel("dN")
		if save_dir is not None:
			plt.savefig(save_dir+"x=t_y=dX.png")
		plt.show()



