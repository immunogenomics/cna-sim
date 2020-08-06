# Import Package Dependencies
import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation
import pandas as pd
from scipy import stats
from methods import methods

# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--causal-clustering', type=str)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# Read Data                                                                                                                                                   
data = sc.read(paths.tbru_h5ad + args.dset + '.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# Simulate Phenotype
np.random.seed(args.index)

nclusters = len(data.obs[args.causal_clustering].unique())
clusters = [args.causal_clustering+'_'+str(i) for i in range(nclusters)]

# Select Causal Clusters on Which to Perform PCA
max_clusters_tested = 10
if nclusters > 10:
    clusters = np.array(clusters)[np.arange(10)]
    nclusters = 10
clusters_fulllabel = clusters
clusters = [i.split("_",1)[1] for i in clusters]

causal_clust = clusters[0] # iterate through                                                                                                                 
cells_in_clust = np.where(data.obs[args.causal_clustering].values==causal_clust)[0]
res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = 20)
df = pd.DataFrame(res)
df['id'] = data.obs['id'].values[cells_in_clust]
Ys = df.groupby('id')[0].aggregate(np.mean)[None,:]

for causal_clust in np.array(clusters)[np.arange(1, len(clusters))]:
    cells_in_clust = np.where(data.obs[args.causal_clustering].values==causal_clust)[0]
    res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = 20)
    df = pd.DataFrame(res)
    df['id'] = data.obs['id'].values[cells_in_clust]
    Ys = np.concatenate((Ys,df.groupby('id')[0].aggregate(np.mean)[None,:]), axis=0)

# Impute missing values as zero
for i in np.arange(Ys.shape[0]):
    loc_nans = np.isnan(Ys[i,])
    Ys[i,loc_nans] = 0

# Add Noise
Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# Execute Analysis                                                                                                                                           
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None, # no sample-level covariates
    None) # no cellular covariates
res['phenotype'] = clusters_fulllabel

# Write Results to Output File(s)                                                                                                   
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
