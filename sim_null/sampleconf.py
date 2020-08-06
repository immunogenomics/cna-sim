import pickle, argparse
import numpy as np
import scanpy as sc
import mcsc as mc
import paths, simulation

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--nsim', type=int, help='number of replicates to simulate')
parser.add_argument('--index', type=int)
parser.add_argument('--phenotype')
parser.add_argument('--method')
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# read data
data = sc.read(paths.simdata + args.dset + '.h5ad')
sampleXmeta = data.uns['sampleXmeta']

# simulate phenotype
np.random.seed(args.index)
Ys = mc.tl._stats.conditional_permutation(
    sampleXmeta.batch.values,
    sampleXmeta[args.phenotype].values,
    args.nsim).T

# do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None,
    None)

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
