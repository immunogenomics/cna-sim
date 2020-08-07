# Import Package Dependencies
import pickle, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import paths, simulation
from methods import methods
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
data = sc.read(paths.tbru_h5ad + args.dset + '.h5ad', backed = "r")
sampleXmeta = data.uns['sampleXmeta']

# Simulate phenotype
np.random.seed(args.index)
n_phenotypes = 20 #Number of PFM PCs to test
sim_pcs = np.arange(n_phenotypes)
pheno_names = ['PFM' + s for s in sim_pcs.astype('str')]

#TODO: revisit interpretability for PFM pcs. Does this phenotype fit in the cell score model?
true_cell_scores = pd.DataFrame(
    data.uns['sampleXnh_featureXpc'][:,:n_phenotypes], columns=pheno_names)
Ys = pd.DataFrame(data.uns['sampleXnh_sampleXpc'][:,:n_phenotypes], columns=pheno_names,
    index=sampleXmeta.index).T
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
        None,
        true_cell_scores.T) # no cellular covariates
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
    phenotype_Rsq.append(stats.pearsonr(true_phenotype, estimated_phenotype)[0])

res['true_estimated_correlation'] = phenotype_Rsq
res['true_phenotype'] = Ys
    
# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')

