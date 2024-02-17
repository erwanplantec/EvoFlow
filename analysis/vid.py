import jax
import numpy as np

from analysis.genomes import GenomeData

def reconstruct_states(data: GenomeData):
    N = len(data.uPt)
    Ps = []
    As = []
    for i in range(N):
        iP = data.iPt[i]
        P = jax.ops.segment_sum(data.uP, iP)
        A = jax.ops.segment_sum(data.At, iP)
