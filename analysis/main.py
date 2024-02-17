from analysis.genomes import GenomeData, compute_genome_data
from analysis.species import SpeciesData, compute_species_data
from analysis.ea import EAData, compute_ea_data

import os
try:
	import _pickle as pickle #type:ignore
except:
	import pickle
import gzip
from typing import NamedTuple

class RunData(NamedTuple):
	gemome_data: GenomeData
	species_data: SpeciesData
	ea_data: EAData

def compute_run_data(data, save_pth=None):

	genome_data = compute_genome_data(data)
	species_data = compute_species_data(genome_data)
	ea_data = compute_ea_data(genome_data)

	run_data = RunData(gemome_data=genome_data,
				   	   species_data=species_data,
				   	   ea_data=ea_data)

	if save_pth is not None:
		with gzip.GzipFile(save_pth, "wb") as f:
			pickle.dump(run_data, f)

	return run_data