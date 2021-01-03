import pickle, argparse
import numpy as np
import pandas as pd
import cna
import paths, simulation
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

# load dataset
data = cna.read(paths.tbru_h5ad + args.dset +'.h5ad')
sampleXmeta = data.samplem
if args.dset[0:4]=="harm":
    data.obsm['X_pca'] = data.X

# simulate phenotype
np.random.seed(args.index)
n_phenotypes = 20
sim_pcs = np.arange(n_phenotypes)
pheno_names = ["PC" + s for s in sim_pcs.astype("str")]
true_cell_scores = pd.DataFrame(data.obsm['X_pca'][:,:n_phenotypes], columns=pheno_names,
                                index=data.obs.index)
Ys = simulation.avg_within_sample(data, true_cell_scores)
print(Ys.shape)

# add noise
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
