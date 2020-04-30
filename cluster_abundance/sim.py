import pickle, argparse
import numpy as np
import scanpy as sc
import paths, simulation

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--nsim', type=int, help='number of replicates to simulate')
parser.add_argument('--noisesize', type=int,
    help='magnitude of noise; minimal value is 1, corresponding to no noise.')
parser.add_argument('--outof', type=int, help='max possible value of noisesize')
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
np.random.seed(args.noisesize)
Y = sampleXmeta[args.phenotype]
Yvar = np.std(Y)
noiselevels = np.linspace(0, 5*Yvar, args.outof)
noiselevel = noiselevels[args.noisesize-1]
noise = np.random.randn(args.nsim, len(Y)) * noiselevel
Ys = Y[None,:] + noise

# do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    None,
    None)
res['noises'] = np.array([noiselevel]*len(Ys))

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.noisesize) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
