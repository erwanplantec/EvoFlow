import jax
import numpy as np
from typing import NamedTuple
from analysis.genomes import GenomeData
import jax.numpy as jnp

class SpatialData(NamedTuple):
	comPt: list
	bbminPt: list
	bbmaxPt: list
	oPPt: list


def overlap(bbmin1, bbmax1, bbmin2, bbmax2):
	bbmin = np.maximum(bbmin1, bbmin2)
	bbmax = np.minimum(bbmax1, bbmax2)
	return np.prod(bbmax-bbmin)

def compute_spatial_data(data: GenomeData):

	uPit = data.uPit
	iPt = data.iPt

	Xsq = iPt[0].shape[0]
	X = int(np.sqrt(Xsq))
	L = np.mgrid[:X, :X].transpose((1,2,0)).reshape((Xsq, 2))

	bbmins = []
	bbmaxs = []
	coms = []
	ovrlps = []
	for upit, ipt in zip(uPit, iPt):
		n = upit.shape[0]
		com = jax.ops.segment_sum(L, ipt, num_segments=n)
		coms.append(com)
		bbmin = jax.ops.segment_min(L, ipt, num_segments=n)
		bbmins.append(bbmin)
		bbmax = jax.ops.segment_max(L, ipt, num_segments=n)
		bbmaxs.append(bbmax)
		ovrlp = jax.vmap(
			jax.vmap(overlap, in_axes=(None,None,0,0)), in_axes=(0,0, None, None)
		)(bbmin[None], bbmax[None], bbmin[:,None], bbmax[:,None])
		ovrlps.append(ovrlp)


	return SpatialData(comPt=coms, bbminPt=bbmins, bbmaxPt=bbmaxs, 
		               oPPt=ovrlps)

def compute_overlap_matrix(sdata: SpatialData, gdata: GenomeData):
	pass






