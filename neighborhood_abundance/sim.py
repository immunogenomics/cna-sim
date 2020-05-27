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
parser.add_argument('--neighborhood-size', type=int)
parser.add_argument('--num-reps', type=int)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
parser.add_argument('--no-covariates', type=int, default=0)
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# read data
data = sc.read(paths.simdata + args.dset + '.h5ad')
sampleXmeta = data.uns['sampleXmeta']

# simulate phenotype
np.random.seed(args.index)
Ys = np.zeros((args.num_reps, len(sampleXmeta)))

a = data.uns['neighbors']['connectivities']
colsums = np.array(a.sum(axis=0)).flatten()
causalcells = []
for i in range(len(Ys)):
    c = np.random.choice(range(len(data)))
    causalcells.append(c)
    q = np.zeros(len(data))
    q[c] = 1
    for t in range(args.neighborhood_size):
        q = a.dot(q/colsums) + q/colsums
    data.obs['q'] = q
    sampleXmeta['avgq'] = data.obs.groupby('id').q.aggregate(np.mean)*1000
    Ys[i] = sampleXmeta.avgq.values
print(Ys.shape)

Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

# set up covariates if necessary
if args.no_covariates:
    print('NO covariates')
    scovs = None
    ccovs = None
else:
    print('WITH covariates')
    scovs = sampleXmeta[['age', 'Sex_M', 'TB_STATUS_CASE', 'NATad4KR']].values
    ccovs = data.obs[['nUMI','percent_mito']].values

# do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    sampleXmeta.batch.values,
    sampleXmeta.C.values,
    scovs,
    ccovs)
res['phenos'] = np.ones(len(Ys))
res['causalcells'] = np.array(causalcells)

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
