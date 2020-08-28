# Import Package Dependencies
import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation
import pandas as pd

# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--causal-clustering', type=str)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
parser.add_argument('--QC-clusters', type=bool)
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# Read Data                                                                                                                                                   

data = sc.read(paths.tbru_h5ad + args.dset + '.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# Simulate Phenotype

np.random.seed(args.index)

clusters = data.obs[args.causal_clustering].unique()
if args.QC_clusters:
     clusters = simulation.discard_bad_clusters(data, args.causal_clustering,
                                                     min_cells_per_sample = 50,
                                                     min_samples_per_cluster = 10,
                                                     clust_batch_cor_thresh = 0.25)
     clusters = clusters.astype(int)

n_phenotypes = len(clusters)
if n_phenotypes > 10:
    n_phenotypes = 10
    clusters = clusters[0:10]
pheno_names = [args.causal_clustering+'_'+str(i) for i in clusters]

true_cell_scores = np.zeros((n_phenotypes, data.obsm['X_pca'].shape[0]))
for i_phenotype in np.arange(n_phenotypes):
    clust = clusters[i_phenotype]
    cells_in_clust = np.where(data.obs[args.causal_clustering].values==str(clust))[0]
    res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = 3)
    true_cell_scores[i_phenotype, cells_in_clust] = res[:,0]

true_cell_scores = pd.DataFrame(true_cell_scores.T, columns=pheno_names,
                                index=data.obs.index)

Ys = simulation.avg_within_sample(data, true_cell_scores)
print(Ys.shape)

# Add noise                                                                                                                                                                                
Ys = simulation.add_noise(Ys, args.noise_level)

# Execute Analysis                                                                                                                                                                         
res = simulation.simulate(
        args.method,
        data,
        Ys.values,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        None, # no sample-level covariates                                                                                                                                                 
        None, # no cellular covariates                                                                                                                                                     
        true_cell_scores.T,
        False, # do not report per-cell scores
        True) # Filter phenotypes based on correlation to batch
res['phenotype'] = pheno_names

# Write Results to Output File(s)                                                                                                                                                          
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
