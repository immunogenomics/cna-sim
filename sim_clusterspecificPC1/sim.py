import pickle, argparse
import numpy as np
import cna
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
data = cna.read(paths.simdata + args.dset +'.h5ad')
sampleXmeta = data.samplem
if args.dset[0:4]=="harm":
    data.obsm['X_pca'] = data.X

# Simulate Phenotype
np.random.seed(args.index)
clusters = data.obs[args.causal_clustering].unique()
if args.QCclusters:
     clusters = simulation.discard_bad_clusters(data, args.causal_clustering,
                                                     min_cells_per_sample = 50,
                                                     min_samples_per_cluster = 10,
                                                     clust_batch_cor_thresh = 0.25)
     clusters = clusters.astype(int)

if len(clusters) > 10:
    clusters = sorted(clusters)[:10]
PCs_per_cluster = np.ceil(30/len(clusters)).astype(int)
print("PCs per cluster ="+str(PCs_per_cluster))
n_phenotypes = len(clusters)*PCs_per_cluster

true_cell_scores = []
pheno_names = []

for c in clusters:
    print("creating PCs for cluster "+str(c))
    cells_in_clust = data.obs[args.causal_clustering].values == str(c)
    print('\t', cells_in_clust.sum(), 'cells in cluster')
    res = sc.pp.pca(data=data.obsm['X_pca'][cells_in_clust,:], n_comps=PCs_per_cluster)
    for npc in np.arange(PCs_per_cluster):
        pheno = np.zeros(len(data))
        pheno[cells_in_clust] = res[:,npc]
        true_cell_scores.append(pheno)
        pheno_names.append(args.causal_clustering+'_c'+str(c)+'_PC'+str(npc))
true_cell_scores = np.array(true_cell_scores)
true_cell_scores = pd.DataFrame(true_cell_scores.T, columns=pheno_names,
                                index=data.obs.index)
Ys = simulation.avg_within_sample(data, true_cell_scores)
print(Ys.shape)

# Add noise
Ys = simulation.add_noise(Ys, args.noise_level)

# Execute Analysis
for i, res in enumerate(simulation.simulate(
        args.method,
        data,
        Ys.values,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        None, # no sample-level covariates
        None, # no cellular covariates
        true_cell_scores.T,
        report_cell_scores=False,
        QC_phenotypes=False)):

    # add phenotype id
    vars(res)['id'] = str(args.index)+'.'+res.pheno

    # write results
    outfile = '{}{}.{}.p'.format(paths.simresults(args.dset, args.simname), args.index, i)
    pickle.dump(res, open(outfile, 'wb'))
