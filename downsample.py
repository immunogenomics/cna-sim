import pandas as pd
import numpy as np
import scanpy as sc
import mcsc as mc
import paths
import argparse

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--Nbatches', type=int, default=None)
parser.add_argument('--maxNcells', type=int, default=None)
parser.add_argument('--minNcells', type=int, default=None)
parser.add_argument('--propcells', type=float, default=None)
parser.add_argument('--inname', type=str)
parser.add_argument('--outname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

print('\n\n****')
print(args)
print('****\n\n')

np.random.seed(args.seed)

# read in dataset
print('reading')
data = sc.read(paths.tbru_h5ad + args.inname + '.h5ad')
if not mc.pp.issorted(data):
    print('ERROR: data are not sorted by sample/batch')
sampleXmeta = data.uns['sampleXmeta']

# downsample samples
print('downsampling samples')
batches = sampleXmeta.batch.unique()
# remove batch 29, which has a really strong batch effect that MASC likely can't model
batches = [b for b in batches if b != 29]
if args.Nbatches is None:
    args.Nbatches = len(batches)
batches_ = np.random.choice(batches, replace=False, size=args.Nbatches)
sampleXmeta_ = sampleXmeta[sampleXmeta.batch.isin(batches_)].copy()
print('N =', len(sampleXmeta_), 'before minNcells')
if args.minNcells is not None:
    sampleXmeta_ = sampleXmeta_[sampleXmeta_.C >= args.minNcells]
N = len(sampleXmeta_)
print('N =', N, 'after minNcells')
data_ = data[data.obs.id.isin(sampleXmeta_.index.values)]

# downsample cells
print('downsampling cells')
if args.maxNcells is not None:
    indices = np.concatenate([[0], np.cumsum(sampleXmeta_.C.values)])
    keep = np.concatenate([
        np.random.choice(np.arange(i, j), replace=False, size=min(j-i, args.maxNcells))
        for i,j in zip(indices[:-1], indices[1:])
        ])
    data_ = data_[keep]
    sampleXmeta_.C = mc.pp.sample_size(data_)
elif args.propcells is not None:
    indices = np.concatenate([[0], np.cumsum(sampleXmeta_.C.values)])
    keep = np.concatenate([
        np.random.choice(np.arange(i, j), replace=False, size=int(args.propcells*(j-i)))
        for i,j in zip(indices[:-1], indices[1:])
        ])
    data_ = data_[keep]
    sampleXmeta_.C = mc.pp.sample_size(data_)

print('data shape', data_.shape)

# assign sampleXmeta and remove other metadata
data_.uns['sampleXmeta'] = sampleXmeta_
data_.obsm.pop('X_pca', None)
data_.obsm.pop('X_umap', None)
data_.uns.pop('neighbors', None)
print(data_)

# sanity check
if not mc.pp.issorted(data_):
    print('ERROR: data are not sorted by sample')

# compute nearest neighbor graph
print('computing nn graph')
sc.pp.neighbors(data_)

# computer neighborhood abundances and PCA of that
print('computing neighborhood abundances and PCA')
mc.tl._newdiff.nns(data_)
mc.tl._newdiff.pca(data_)

# clustering at different resolutions
for res in [0.2, 1, 2, 5]:
    print('clustering at resolution', res)
    n = 'dleiden'+str(res).replace('.','p')
    sc.tl.leiden(data_, resolution=res, key_added=n)
    print('\t', len(data_.obs[n].unique()), 'clusters')
    for i in sorted(data_.obs[n].unique()):
        c = data_.obs.groupby(by='id')[n].aggregate(lambda x: np.mean(x==str(i)))
        data_.uns['sampleXmeta'][n+'_'+str(i)] = c

# UMAP
print('computing umap')
sc.tl.umap(data_)

# write
if args.outname is None:
    args.outname = args.inname + '.N=' + str(N)
print('writing', args.outname)
data_.write(paths.simdata + args.outname + '.h5ad')
print('done')
