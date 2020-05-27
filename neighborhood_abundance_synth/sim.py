import pickle, argparse
import numpy as np
import scanpy as sc
import anndata as ad
import paths, simulation

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--simname')
parser.add_argument('--dset')
parser.add_argument('--method')
parser.add_argument('--index', type=int)
parser.add_argument('--neighborhood-size', type=int)
parser.add_argument('--num-reps', type=int)
parser.add_argument('--noise-level', type=float) #in units of std dev of noiseless phenotype
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

# simulate dataset
np.random.seed(0)
N = 100
G = 100
C = 50
pc1 = np.ones(G)
samplemeans = 0.5*np.random.randn(N)
X = np.random.randn(C*N, G) + np.outer(np.repeat(samplemeans, C), pc1)
data = ad.AnnData(X=X)
data.obs['id'] = np.repeat(np.arange(N), C)
sc.tl.pca(data, n_comps=20)
sc.pp.neighbors(data)
sc.tl.leiden(data, resolution=1, key_added='leiden1')
C = np.repeat(C, N)
B = np.repeat(np.arange(N/5), 5)[np.argsort(np.random.randn(N))]

# simulate phenotype
np.random.seed(args.index)
Ys = np.zeros((args.num_reps, N))

a = data.uns['neighbors']['connectivities']
colsums = np.array(a.sum(axis=0)).flatten() + 1
causalcells = []
for i in range(len(Ys)):
    c = np.random.choice(range(len(data)))
    causalcells.append(c)
    q = np.zeros(len(data))
    q[c] = 1
    for t in range(args.neighborhood_size):
        q = a.dot(q/colsums) + q/colsums
    data.obs['q'] = q
    Ys[i] = data.obs.groupby('id').q.aggregate(np.mean)*1000
print(Ys.shape)

Yvar = np.std(Ys, axis=1)
noiselevels = args.noise_level * Yvar
noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
Ys = Ys + noise

scovs = None
ccovs = None

# do analysis
res = simulation.simulate(
    args.method,
    data,
    Ys,
    B,
    C,
    scovs,
    ccovs)
res['phenos'] = np.ones(len(Ys))
res['causalcells'] = np.array(causalcells)

# write results
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
