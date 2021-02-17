import pickle, argparse
import numpy as np
import pandas as pd
import cna
import paths, simulation

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--nsim', type=int, help='number of replicates to simulate')
parser.add_argument('--index', type=int)
parser.add_argument('--method')
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# read data
data = cna.read(paths.tbru_h5ad + args.dset + '.h5ad')
sampleXmeta = data.samplem

# simulate phenotype
np.random.seed(args.index)
causal_batch = np.random.choice(sampleXmeta.batch.unique(), replace=True, size=args.nsim)
Ys = np.array([
        (sampleXmeta.batch.values == cb).astype(np.float64)
        for cb in causal_batch])
Ys += 0.01 * np.random.randn(args.nsim, len(sampleXmeta))

# do analysis
true_cell_scores = pd.DataFrame(np.random.randn(len(data), args.nsim), # this is a dummy value
                    columns=['batch'+str(cb)+','+str(i) for i, cb in enumerate(causal_batch)])
for i, res in enumerate(simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None, #sampleXmeta[sample_covs].values,
    None, #No cell-level covariates
    true_cell_scores.T,
    report_cell_scores=False,
    QC_phenotypes=False)):

    # add phenotype id
    vars(res)['id'] = str(args.index) + '.' + res.pheno

    # write results
    outfile = '{}{}.{}.p'.format(paths.simresults(args.dset, args.simname), args.index, i)
    pickle.dump(res, open(outfile, 'wb'))
