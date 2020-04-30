import pandas as pd
import numpy as np
import scanpy as sc
import mcsc as mc
import paths
import argparse

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--Nbatches', type=int)
parser.add_argument('--maxNcells', type=int, default=None)
parser.add_argument('--inname', type=str)
parser.add_argument('--outname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

# read in dataset
print('reading')
data = sc.read(paths.tbru_h5ad + args.inname + '.h5ad')
if not mc.pp.issorted(data):
    print('ERROR: data are not sorted by sample/batch')
sampleXmeta = data.uns['sampleXmeta']

# downsample
print('downsampling')
batches = sampleXmeta.batch.unique()
batches_ = np.random.choice(batches, size=args.Nbatches)
sampleXmeta_ = sampleXmeta[sampleXmeta.batch.isin(batches_)]
N = len(sampleXmeta_)
print('N =', N)
data_ = data[data.obs.id.isin(sampleXmeta_.index.values)]
data_.uns['sampleXmeta'] = sampleXmeta_
if not mc.pp.issorted(data_):
    print('ERROR: data are not sorted by sample/batch')
#TODO: if maxNcells is not None, downsample each sample's cells to the right number

# compute nearest neighbor graph
print('computing nn graph')
sc.pp.neighbors(data_)

print('clustering')
sc.tl.leiden(data_)
for i in sorted(data_.obs.leiden.unique())[:1]:
    c = data_.obs.groupby(by='id')['leiden'].aggregate(lambda x: np.mean(x==str(i)))
    data_.uns['sampleXmeta']['c'+str(i)] = c

# write
if args.outname is None:
    args.outname = args.inname + '.N=' + str(N)
print('writing', args.outname)
data_.write(paths.simdata + args.outname + '.h5ad')
print('done')
