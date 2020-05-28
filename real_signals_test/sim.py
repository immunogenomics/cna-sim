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
phenotypes = ['Weight','NATad4KR','TB_STATUS_CASE', 'Sex_M', 'season_Winter']

all_Ys = sampleXmeta[phenotypes].values.T
print(all_Ys.shape)

# Add noise                                                                                                                                                       
all_Yvar = np.std(all_Ys, axis=1)
noiselevels = args.noise_level * all_Yvar
noise = np.random.randn(*all_Ys.shape) * noiselevels[:,None]
all_Ys = all_Ys + noise

# Execute Analysis
res = None
for n_phen in np.arange(len(phenotypes)):
    Ys = all_Ys[n_phen,:].reshape(1,all_Ys.shape[1])
    covar_names = phenotypes.copy()
    covar_names.pop(n_phen)
    covar_names.append('age')
    covar_data = sampleXmeta[covar_names].values
    res_new = simulation.simulate(
        args.method,
        data,
        Ys,
        sampleXmeta.batch.values,
        sampleXmeta.C.values,
        covar_data,
        None) # DO NOT CURRENTLY ACCOUNT FOR CELLULAR COVARS 
    if res is None:
        res = res_new
    else:
        res['zs'] = np.vstack((res['zs'], res_new['zs']))
        res['fwers'] = np.vstack((res['fwers'], res_new['fwers']))
        res['ntests'] = np.append(res['ntests'],res_new['ntests'])
        res['others'] = np.append(res['others'],res_new['others'])  
res['phenotypes'] = phenotypes

# Write Results to Output File(s)
outfile = paths.simresults(args.dset, args.simname) + str(args.index) + '.p'
print('writing', outfile)
pickle.dump(res, open(outfile, 'wb'))
print('done')
