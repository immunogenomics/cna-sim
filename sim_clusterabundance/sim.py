import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--causal-clustering', type=str)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
parser.add_argument('--no-covariates', type=int, default=0)
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# Read Data
data = sc.read(paths.tbru_h5ad + args.dset + '.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# Sample-level Covariates
sample_covs = ['age', 'Sex_M', 'TB_STATUS_CASE', 'season_Winter', 'Weight', 'NATad4KR']

# Simulate Phenotype
np.random.seed(args.index)
nclusters = len(data.obs[args.causal_clustering].unique())
clusters = [args.causal_clustering+'_'+str(i) for i in range(nclusters)]
Ys = sampleXmeta[clusters].values.T

# Exclude small clusters
sizes = data.obs[args.causal_clustering].value_counts()
big = (sizes >= 1000).sort_index().values
Ys = Ys[big]

print(big)
print([c for b, c in zip(big, clusters) if b])
print(Ys.shape)

Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# Do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None, #sampleXmeta[sample_covs].values,
    None)
res['clusterids'] = np.arange(nclusters)[big]

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
