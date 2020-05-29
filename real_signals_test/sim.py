# Import Package Dependencies
import pickle, argparse, os
import numpy as np
import scanpy as sc
import paths, simulation

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dset')
parser.add_argument('--simname')
parser.add_argument('--method', nargs='+')
parser.add_argument('--index', type=int)
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
phenotypes = ['Weight','NATad4KR','TB_STATUS_CASE','Sex_M','season_Winter','age']

all_Ys = sampleXmeta[phenotypes].values.T
print(all_Ys.shape)

# Add noise
all_Yvar = np.std(all_Ys, axis=1)
noiselevels = args.noise_level * all_Yvar
noise = np.random.randn(*all_Ys.shape) * noiselevels[:,None]
all_Ys = all_Ys + noise

scovs = np.array([
    sampleXmeta[[p_ for p_ in phenotypes if p_ != p]].values
    for p in phenotypes])

# Execute Analysis
for m in args.method:
    res = simulation.simulate(
        m,
        data,
        all_Ys,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        scovs,
        None) # DO NOT CURRENTLY ACCOUNT FOR CELLULAR COVARS
    res['phenotypes'] = np.array(phenotypes)

    # Write Results to Output File(s)
    outdir = paths.simresults(args.dset, args.simname) + m + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = outdir + str(args.index) + '.p'
    print('writing', outfile)
    pickle.dump(res, open(outfile, 'wb'))
    print('done')
