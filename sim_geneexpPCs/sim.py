# Import Package Dependencies
import pickle, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import paths, simulation
from scipy import stats

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# Read Data
data = sc.read(paths.tbru_h5ad + args.dset +'.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# simulate phenotype
np.random.seed(args.index)
n_phenotypes = 20
sim_pcs = np.arange(n_phenotypes)
pheno_names = ["causal_PC" + s for s in sim_pcs.astype("str")]

true_cell_scores = pd.DataFrame(data.obsm['X_pca'][:,:n_phenotypes], columns=pheno_names)
Ys = simulation.avg_within_sample(data, true_cell_scores)
print(Ys.shape)

# Add noise
Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# Execute Analysis
res = simulation.simulate(
        args.method,
        data,
        Ys.values,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        None, # no sample-level covariates
        None, # no cellular covariates
        true_cell_scores.T)
res['phenotype'] = pheno_names

# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')

