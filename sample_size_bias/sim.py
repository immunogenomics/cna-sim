import pickle, argparse
import pandas as pd
import numpy as np
import cna
import paths, simulation
import multianndata as mad
import scanpy as sc
from methods import methods
np.random.seed(0)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--small_pop_frac', type = float, default = 0.1)
parser.add_argument('--C', type = int, default = 300) # Num cells per case sample
parser.add_argument('--method', type = str, default = "CNA")
parser.add_argument('--N', type=float, default = 50) # Num samples
parser.add_argument('--G', type =int, default = 50) # Num genes
parser.add_argument('--noise', type = float, default=1) #Noise
parser.add_argument('--n_trials', type = int, default = 50)
parser.add_argument('--outfile_name',type = str)
args = parser.parse_args()
print('\n\n****')
print(args)
print('****\n\n')

use_weighted_pca = False
uncorrelated_downsample = False
npcs_retain = 20
remaining_pop_frac = round((1-args.small_pop_frac)/2,2)

# Assign covariates to samples
# Covs is samples x covs; 1 if covariate is true of that sample, 0 o.w.
# Male/non-male is balanced within cases and within controls
covs = pd.DataFrame(index=pd.Series(np.arange(args.N), name='id', dtype=int))
covs['case'] = [0]*int(args.N/2) + [1]*int(args.N/2)
covs['male'] = [0]*int(2*args.N/8) + [1]*int(2*args.N/8) + [0]*int(2*args.N/8) + [1]*(args.N - int(2*args.N/8)-int(2*args.N/8)-int(2*args.N/8))
covs['baseline'] = 1

# Define GE profile of cell populations
# Three cell populations (rows)
# First 1/4 of genes same (define first population, noise added later)
# Second 1/4 of genes same, define second population of cells
# Last 1/2 of genes same, define last population of cells
H = np.zeros((3, args.G))
H[0,:int(args.G/2)] = 1
H[1,int(args.G/2):] = 1
H[2,:int(args.G/2)] = 1; H[2,:int(args.G/4)] = 2

# Define cell ids for the cells in each sample
# Props is 1 x n-1 dimensional where n is the number of cell populations (3)
# Gives proportion of cells in this sample that should be from each population
# Final output W is C cells x 3 pops, contains 1 to identify pop each cell came from
#      and is 0 o.w.

def getW(props):
    cell_ids = np.concatenate([np.array([i]*int(p*args.C)) for i,p in enumerate(props)])
    cell_ids = np.concatenate([cell_ids, np.array([len(props)]*int(args.C-len(cell_ids)))])
    W = np.zeros((int(args.C), len(props) + 1))
    for i in range(len(props) + 1):
        W[np.where(cell_ids == i)[0], i] = 1
    return W

props = np.array([
    [0, 0], 
# Case: difference from baseline in frac of cells from pop 1, pop 2 
#(pop 3 delta computed to sum to 1)
    [0, 0], 
# Male: difference from baseline in frac of cells from pop 1, pop 2 
#(pop 3 delta computed to sum to 1)
    [args.small_pop_frac, remaining_pop_frac]  
# Baseline frac of cells from pop 1, pop 2 (1-sum of these is frac from pop 3)
])

all_frac_sig = []
for frac_retain in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]: # fraction of cells to retain in control samples, 0.15, 0.2, 0.25, 0.3, 0.5
    print("Fraction of case sample size retained in control samples: "+str(frac_retain))
    trials_ps = []
    for i in np.arange(args.n_trials):
        Ws = np.array([getW(c.dot(props)) for _, c in covs.iterrows()])

        # Downsample cells in each control sample
        # First cell always included
        Ws = [Ws[i][np.concatenate(([True], np.random.choice([True, False], size=Ws[i].shape[0]-1, 
                         p=[frac_retain, 1-frac_retain]))),:] if covs['case'][i]==1 else Ws[i] \
                          for i in np.arange(len(Ws))]
        
        # Get 'true' cluster assignments
        cluster_assn = np.concatenate([W.dot(np.array([[0],[1],[2]])) for W in Ws])
        cluster_assn = [str(int(cluster_assn[i])) for i in np.arange(len(cluster_assn))]

        # Concatenate gene expression x cells matrices per sample together
        # This is the cells x genes expression matrix (the single-cell data)
        X = np.concatenate([W.dot(H) + args.noise*np.random.randn(W.shape[0], args.G) for W in Ws])
        sampleids = np.concatenate([np.repeat(covs.index[i], Ws[i].shape[0]) for i in np.arange(len(Ws))])
        covs_C = np.concatenate([np.array([W.shape[0]]) for W in Ws])

        # create multianndata object
        covs['C'] = covs_C
        d = mad.MultiAnnData(X=X,
                             obs=pd.DataFrame({'id':sampleids, 'cluster_assn':cluster_assn}),
                             samplem=covs.drop(columns=['baseline']))
        d.samplem['batch'] = np.tile(range(5), int(args.N/5))
        d.var = pd.DataFrame({'gene':['gene'+str(i) for i in range(args.G)]})
        
        if use_weighted_pca:
            # weighted scaling
            cell_weights = np.ones(d.X.shape[0]).reshape(1,-1) # balanced weight case
            gene_weightedmeans = np.dot(cell_weights, d.X)/(d.X.shape[0]-1)
            gene_weightedvariance = np.dot(cell_weights, np.square(d.X-gene_weightedmeans))/(d.X.shape[0]-1)
            scaled_mat = (d.X-gene_weightedmeans)/gene_weightedvariance

            U, svs, UT = np.linalg.svd(scaled_mat.dot(scaled_mat.T))
            V = scaled_mat.T.dot(U) / np.sqrt(svs)
            d.obsm['X_weightedpca'] = U[:,:npcs_retain]
            sc.pp.neighbors(d, use_rep = 'X_weightedpca')
        else:
            sc.pp.neighbors(d)
        
        if args.method =="CNA":
            # perform association test for case/ctrl status, controlling for sex as a covariate and accounting for potential batch effect
            res = cna.tl.association(d,                   #dataset
                d.samplem.case,                   #sample-level attribute of intests (case/control status)
                covs=None,         #covariates to control for (in this case just one)
                batches=d.samplem.batch,          #batch ids for each sample so that cna can account for batch effect
                local_test=True)
            new_p = res.p
        else:
            sc.tl.leiden(d, resolution=1.1)
            print(d.obs.groupby(by=["cluster_assn", "leiden"]).count())
            
            # perform association test for case/ctrl status, accounting for potential batch effect (assumes no covariate effects)
            res = methods._MASC(data = d, 
                            Y = d.samplem.case.values, 
                            B = d.samplem.batch.values, 
                            C =d.samplem.C.values, #Fix
                            T = None, # sample-level covariates                                                                                                                                                      
                            s = None, # Cell-level covariates
                            clustertype='leiden')
            masc_p = res[3][1]
            new_p = np.min(masc_p)*len(masc_p)
        covs = covs.drop(columns='C')

        trials_ps.append(new_p)
    all_frac_sig.append(np.sum(np.array(trials_ps)<0.05)/len(trials_ps))
    
outfile = open(args.outfile_name,'wb')
pickle.dump(all_frac_sig, outfile)
outfile.close()
