# Import Package Dependencies
import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation

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
    
# Define covariates
sample_covs = ['age', 'Sex_M', 'TB_STATUS_CASE', 'season_Winter', 'Weight', 'NATad4KR']

# simulate phenotype
np.random.seed(args.index)
n_sim_pcs = 20 #nclusters
sim_pcs = np.arange(n_sim_pcs)+1
sim_pc_names = ["causal_PC" + s for s in sim_pcs.astype("str")] #clusters

for n_pc in sim_pcs:
    data.obs['causal_PC'+str(n_pc)] = data.obsm['X_pca'][:,n_pc-1]
    sampleXmeta['causal_PC'+str(n_pc)] = data.obs.groupby('id')['causal_PC'+str(n_pc)].aggregate(np.mean)

Ys = sampleXmeta[sim_pc_names].values.T
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
        Ys,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        None, #sampleXmeta[['age', 'Sex_M', 'TB_STATUS_CASE', 'NATad4KR']].values,
        None) # no cellular covariates
res['phenotype'] = np.arange(n_sim_pcs)
    
# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')

