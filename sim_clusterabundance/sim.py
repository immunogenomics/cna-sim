import pickle, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import paths, simulation

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--causal-clustering', type=str)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
parser.add_argument('--QC-clusters', type=bool, default=0)
args = parser.parse_args()
print('\n\n****')
print(args)
print('****\n\n')

# Read Data
data = sc.read(paths.tbru_h5ad + args.dset + '.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# Simulate Phenotype
np.random.seed(args.index)

## compute true cell scores
true_cell_scores = pd.get_dummies(data.obs[args.causal_clustering])
if args.QC_clusters:
     retain_clusters = simulation.discard_bad_clusters(data, args.causal_clustering, 
                                                     min_cells_per_sample = 50, 
                                                     min_samples_per_cluster = 10,
                                                     clust_batch_cor_thresh = 0.25)
     true_cell_scores = true_cell_scores.iloc[:,np.where([name in retain_clusters for name in true_cell_scores.columns.values.astype(str)])[0]]
print(true_cell_scores.shape)
true_cell_scores.columns = true_cell_scores.columns.values.astype(int)
true_cell_scores.rename(columns={c:'cluster'+str(c) for c in true_cell_scores.columns},
                            inplace=True)
pheno_names = true_cell_scores.columns
Ys = simulation.avg_within_sample(data, true_cell_scores)
print(Ys.shape)

# Add noise
Ys = simulation.add_noise(Ys, args.noise_level)

# Do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys.values,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None, #sampleXmeta[sample_covs].values,
    None,
    true_cell_scores.T,
    False, # Do not report cell scores
    True) # Filter phenotypes correlated with batch
print(Ys.shape)
res['phenotype'] = pheno_names

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
