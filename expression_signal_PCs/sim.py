# Import Package Dependencies
import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation
from methods import methods # For phenotype diffusion function
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
phenotype_names = ["causal_PC" + s for s in sim_pcs.astype("str")]                                           

for i_phenotype in np.arange(n_phenotypes):
    n_pc = sim_pcs[i_phenotype]
    data.obs[phenotype_names[i_phenotype]] = data.obsm['X_pca'][:,n_pc]
    sampleXmeta[phenotype_names[i_phenotype]] = data.obs.groupby('id')[phenotype_names[i_phenotype]].aggregate(np.mean)

Ys = sampleXmeta[phenotype_names].values.T
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
        None, # no sample-level covariates
        None) # no cellular covariates
res['phenotype'] = phenotype_names

# Assess correlation of true and estimated neighborhood values
phenotype_Rsq = list()
n_pfm_pcs = 20
for i_phenotype in np.arange(n_phenotypes):
    phenotype_name = phenotype_names[i_phenotype]
    true_phenotype = data.obs[phenotype_name].values
    if args.method=="mixedmodel_nfm_npcs20":
        beta_vals = res['beta_vals'][i_phenotype]
        estimated_phenotype = np.sum(data.uns['sampleXnh_featureXpc'][:,0:n_pfm_pcs]*beta_vals.reshape(1,n_pfm_pcs), axis = 1)
    else:
        cluster_betas = -1*res['beta_vals'][i_phenotype,:] # Flip sign based on current version of runmasc.R
        estimated_phenotype = np.zeros(data.obs.shape[0])
        for i_cluster in np.arange(len(cluster_betas)):
            cluster_cells = data.obs[args.method.split("_")[1]]==str(i_cluster) # args.method
            estimated_phenotype[cluster_cells]=np.repeat(cluster_betas[i_cluster], np.sum(cluster_cells))
    estimated_phenotype = methods.diffuse_phenotype(data, estimated_phenotype)
    phenotype_Rsq.append(stats.pearsonr(true_phenotype, estimated_phenotype)[0])

res['true_estimated_correlation'] = phenotype_Rsq
res['true_phenotype'] = Ys
    
# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')

