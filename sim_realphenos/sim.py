# Import Package Dependencies
import pickle, argparse, os
import numpy as np
import cna
import pandas as pd
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
data = cna.read(paths.simdata + args.dset + '.h5ad')
sampleXmeta = data.samplem
sampleXmeta['age2'] = sampleXmeta.age**2

# simulate phenotype
np.random.seed(args.index)
phenos = ['TB_STATUS_CASE','EURad4KR','Sex_M','season_Winter','age']
covs = {
    'TB_STATUS_CASE':['age','age2','Sex_M','season_Winter','EURad4KR'],
    'EURad4KR':['Height','Weight','edu_cat_belowHighschool','season_Winter','TB_STATUS_CASE'],
    'Sex_M':['Height','num_scar','TB_STATUS_CASE'],
    'season_Winter':['age','Height','EURad4KR','BCG_scar','season_Spring'],
    'age':['Weight','num_scar','season_Winter','TB_STATUS_CASE'],
}

Ys = sampleXmeta[phenos].T
print(phenos)
print(Ys.shape)

# Add noise
Ys = simulation.add_noise(Ys, args.noise_level)

# Execute Analysis
for i, (pheno, Y) in enumerate(Ys.iterrows()):
    res = next(
        simulation.simulate(
            args.method,
            data,
            Y.values.reshape((1,-1)),
            sampleXmeta.batch.values,
            sampleXmeta.C.values,
            sampleXmeta[covs[pheno]].values,
            None, # DO NOT CURRENTLY ACCOUNT FOR CELLULAR COVARS
            pd.DataFrame(np.zeros((1,len(data.obs))), index=[pheno]),
            report_cell_scores=False,
            QC_phenotypes=False)
        )

    # add pheno id
    vars(res)['id'] = str(args.index)+'.'+res.pheno

    # Write Results to Output File(s)
    outfile = '{}{}.{}.p'.format(paths.simresults(args.dset, args.simname), args.index, i)
    pickle.dump(res, open(outfile, 'wb'))
