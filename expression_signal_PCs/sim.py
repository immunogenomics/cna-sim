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
parser.add_argument('--causal-pc-set', type=str)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# Read Data
data = sc.read(paths.simdata + args.dset + '.h5ad')
sampleXmeta = data.uns['sampleXmeta']

# simulate phenotype                                                                                                                                              
np.random.seed(args.index)
nclusters = 20
clusters = ['causal_PC1','causal_PC2', 'causal_PC3', 'causal_PC4', 'causal_PC5',\
                'causal_PC6', 'causal_PC7', 'causal_PC8', 'causal_PC9','causal_PC10',\
                'causal_PC11','causal_PC12', 'causal_PC13', 'causal_PC14', 'causal_PC15',
            'causal_PC16', 'causal_PC17', 'causal_PC18', 'causal_PC19', 'causal_PC20']

data.obs['causal_PC1'] = data.obsm['X_pca'][:,0]
data.obs['causal_PC2'] = data.obsm['X_pca'][:,1]
data.obs['causal_PC3'] = data.obsm['X_pca'][:,2]
data.obs['causal_PC4'] = data.obsm['X_pca'][:,3]
data.obs['causal_PC5'] = data.obsm['X_pca'][:,4]
data.obs['causal_PC6'] = data.obsm['X_pca'][:,5]
data.obs['causal_PC7'] = data.obsm['X_pca'][:,6]
data.obs['causal_PC8'] = data.obsm['X_pca'][:,7]
data.obs['causal_PC9'] = data.obsm['X_pca'][:,8]
data.obs['causal_PC10'] = data.obsm['X_pca'][:,9]
data.obs['causal_PC11'] = data.obsm['X_pca'][:,10]
data.obs['causal_PC12'] = data.obsm['X_pca'][:,11]
data.obs['causal_PC13'] = data.obsm['X_pca'][:,12]
data.obs['causal_PC14'] = data.obsm['X_pca'][:,13]
data.obs['causal_PC15'] = data.obsm['X_pca'][:,14]
data.obs['causal_PC16'] = data.obsm['X_pca'][:,15]
data.obs['causal_PC17'] = data.obsm['X_pca'][:,16]
data.obs['causal_PC18'] = data.obsm['X_pca'][:,17]
data.obs['causal_PC19'] = data.obsm['X_pca'][:,18]
data.obs['causal_PC20'] = data.obsm['X_pca'][:,19]
sampleXmeta['causal_PC1'] = data.obs.groupby('id')['causal_PC1'].aggregate(np.mean)
sampleXmeta['causal_PC2'] = data.obs.groupby('id')['causal_PC2'].aggregate(np.mean)
sampleXmeta['causal_PC3'] = data.obs.groupby('id')['causal_PC3'].aggregate(np.mean)
sampleXmeta['causal_PC4'] = data.obs.groupby('id')['causal_PC4'].aggregate(np.mean)
sampleXmeta['causal_PC5'] = data.obs.groupby('id')['causal_PC5'].aggregate(np.mean)
sampleXmeta['causal_PC6'] = data.obs.groupby('id')['causal_PC6'].aggregate(np.mean)
sampleXmeta['causal_PC7'] = data.obs.groupby('id')['causal_PC7'].aggregate(np.mean)
sampleXmeta['causal_PC8'] = data.obs.groupby('id')['causal_PC8'].aggregate(np.mean)
sampleXmeta['causal_PC9'] = data.obs.groupby('id')['causal_PC9'].aggregate(np.mean)
sampleXmeta['causal_PC10'] = data.obs.groupby('id')['causal_PC10'].aggregate(np.mean)
sampleXmeta['causal_PC11'] = data.obs.groupby('id')['causal_PC11'].aggregate(np.mean)
sampleXmeta['causal_PC12'] = data.obs.groupby('id')['causal_PC12'].aggregate(np.mean)
sampleXmeta['causal_PC13'] = data.obs.groupby('id')['causal_PC13'].aggregate(np.mean)
sampleXmeta['causal_PC14'] = data.obs.groupby('id')['causal_PC14'].aggregate(np.mean)
sampleXmeta['causal_PC15'] = data.obs.groupby('id')['causal_PC15'].aggregate(np.mean)
sampleXmeta['causal_PC16'] = data.obs.groupby('id')['causal_PC16'].aggregate(np.mean)
sampleXmeta['causal_PC17'] = data.obs.groupby('id')['causal_PC17'].aggregate(np.mean)
sampleXmeta['causal_PC18'] = data.obs.groupby('id')['causal_PC18'].aggregate(np.mean)
sampleXmeta['causal_PC19'] = data.obs.groupby('id')['causal_PC19'].aggregate(np.mean)
sampleXmeta['causal_PC20'] = data.obs.groupby('id')['causal_PC20'].aggregate(np.mean)

Ys = sampleXmeta[clusters].values.T
print(Ys.shape)

# Add noise
Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# average cellular confounders                                                                                                                                    
conf = ['nUMI','percent_mito']
sampleXmeta[conf] = data.obs.groupby('id')[conf].aggregate(np.mean)

# Execute Analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    sampleXmeta[['age', 'Sex_M', 'TB_STATUS_CASE', 'NATad4KR']].values,
    data.obs[['nUMI','percent_mito']].values)
res['clusterids'] = np.arange(nclusters)

# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
