import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation
import pandas as pd

# argument parsing
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

# read data
data = sc.read(paths.simdata + args.dset + '.h5ad')
sampleXmeta = data.uns['sampleXmeta']

# simulate phenotype
np.random.seed(args.index)
nclusters = len(data.obs[args.causal_clustering].unique())
clusters = [args.causal_clustering+'_'+str(i) for i in range(nclusters)]

causal_clust = 0 # iterate through
cells_in_clust = np.where(data.obs[args.causal_clustering].values.astype('int')==causal_clust)[0]
res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = 20)
df = pd.DataFrame(res)
df['id'] = data.obs['id'].values[cells_in_clust]
Ys = df.groupby('id')[0].aggregate(np.mean)[None,:]

for causal_clust in np.arange(1,nclusters):
    causal_clust = 0 # iterate through
    cells_in_clust = np.where(data.obs[args.causal_clustering].values.astype('int')==causal_clust)[0]
    res = sc.pp.pca(data = data.obsm['X_pca'][cells_in_clust,:], n_comps = 20)
    df = pd.DataFrame(res)
    df['id'] = data.obs['id'].values[cells_in_clust]
    Ys = np.concatenate((Ys,df.groupby('id')[0].aggregate(np.mean)[None,:]), axis=0) 

# Add noise
Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# average cellular confounders
conf = ['nUMI','percent_mito']
sampleXmeta[conf] = data.obs.groupby('id')[conf].aggregate(np.mean)

# do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    sampleXmeta[['age', 'Sex_M', 'TB_STATUS_CASE', 'NATad4KR']].values,
    data.obs[['nUMI','percent_mito']].values)
res['clusterids'] = np.arange(nclusters)

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
