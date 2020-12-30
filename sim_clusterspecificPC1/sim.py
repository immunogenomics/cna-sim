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
parser.add_argument('--QCclusters', type=bool)
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

### Simulate Phenotype                                                                                                                                                                         
# load dataset                                                                                                                                                                    
data = sc.read(paths.tbru_h5ad + args.dset +'.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']
if args.dset[0:4]=="harm":
    data.obsm['X_pca'] = data.X

# for CNAv1                                                                                                                                                                       
if args.method =="mixedmodel_nfm_npcs20":
    data.uns['sampleXnh_featureXpc'] = data.obsm['sampleXnh_featureXpc']

# Simulate Phenotype
np.random.seed(args.index)
clusters = data.obs[args.causal_clustering].unique()
if args.QCclusters:
     clusters = simulation.discard_bad_clusters(data, args.causal_clustering,
                                                     min_cells_per_sample = 50,
                                                     min_samples_per_cluster = 10,
                                                     clust_batch_cor_thresh = 0.25)
     clusters = clusters.astype(int)

n_clusters = len(clusters)
if n_clusters > 10:
    n_clusters = 10
    clusters = clusters[0:10]
PCs_per_cluster = np.ceil(30/len(clusters)).astype(int)
print("PCs per cluster ="+str(PCs_per_cluster))
n_phenotypes = n_clusters*PCs_per_cluster

cluster_names = [args.causal_clustering+'_cluster'+str(i)+"_" for i in clusters]
all_cluster_names = np.repeat(cluster_names, PCs_per_cluster)
all_PC_names = np.tile(np.arange(PCs_per_cluster), n_clusters)
pheno_names = [all_cluster_names[i]+str(all_PC_names[i]) for i in np.arange(len(all_cluster_names))]


true_cell_scores = np.zeros((n_phenotypes, data.obsm['X_pca'].shape[0]))


for i_cluster in np.arange(n_clusters):
    print("creating PCs for cluster "+str(i_cluster))
    clust = clusters[i_cluster]
    cells_in_clust = np.where(data.obs[args.causal_clustering].values==str(clust))[0]
    res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = PCs_per_cluster)
    for npc in np.arange(PCs_per_cluster):
        i_phenotype = (i_cluster*PCs_per_cluster)+npc
        true_cell_scores[i_phenotype, cells_in_clust] = res[:,npc]

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
        True, # do not report per-cell scores
        False) # Filter phenotypes based on correlation to batch
res['phenotype'] = pheno_names

# Write Results to Output File(s)                                                                                                                                                          
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
