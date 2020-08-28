import time, gc
import numpy as np
from methods import methods

def onehot_batch_gen(data):
    num_batches = len(np.unique(data.obs.batch))
    num_cells = len(data.obs.batch)
    onehot_batch = np.zeros((num_cells, num_batches))
    for i_batch in np.arange(len(np.unique(data.obs.batch))):
        batch = np.unique(data.obs.batch)[i_batch]
        cells_in_batch = np.where(data.obs.batch==batch)[0]
        onehot_batch[cells_in_batch,i_batch]=1
    return onehot_batch

def phenos_uncorr_with_batch(data, cell_scores, cor_thresh):
    onehot_batch = onehot_batch_gen(data)
    final_set = []
    for i_pheno in np.arange(cell_scores.shape[1]):
        pheno_cor_wBatch = np.corrcoef(np.hstack((cell_scores[:,i_pheno:(i_pheno+1)], onehot_batch)), rowvar = False)[0,1:-1]
        if np.sum(pheno_cor_wBatch > cor_thresh) ==0:
            final_set.append(i_pheno)
    return final_set

def discard_bad_clusters(data, cluster_res, min_cells_per_sample, min_samples_per_cluster,
                         clust_batch_cor_thresh):
    # Returns the labels for clusters that pass checks for required sample representation
    # and lack of correlation with batch

    # Check clusters for minimum sample representation
    keep_clust = []
    for i_cluster in np.unique(data.obs[cluster_res]):
        cells_in_clust = np.where(data.obs[cluster_res]==i_cluster)[0]
        if (np.sum(data.obs["id"][cells_in_clust].value_counts().values > min_cells_per_sample)>min_samples_per_cluster):
            keep_clust.append(i_cluster)

    # Check retained clusters for correlation with batch
    for i_cluster in keep_clust:
        if i_cluster==keep_clust[0]:
            cluster_scores = (np.array(data.obs[cluster_res]==i_cluster)*1).reshape(-1,1)
        else:
            cluster_scores = np.hstack((cluster_scores, 
                                        (np.array(data.obs[cluster_res]==i_cluster)*1).reshape(-1,1)))
    loc_final_set = phenos_uncorr_with_batch(data,cluster_scores, clust_batch_cor_thresh)
    return np.array(keep_clust)[loc_final_set]

def avg_within_sample(data, true_cell_scores):
    cols = true_cell_scores.columns.values
    true_cell_scores['id'] = data.obs.id

    # this is to use the fact that pandas will match appropriately by sample
    data.uns['sampleXmeta'][cols] = true_cell_scores.groupby('id').aggregate(np.mean)[cols]
    Ys = data.uns['sampleXmeta'][cols]
    data.uns['sampleXmeta'].drop(columns=cols, inplace=True)
    true_cell_scores.drop(columns="id",inplace=True)
    return Ys.T

def add_noise(Ys, noiselevels): #Ys is assumed to have one row per phenotype
    Yvar = np.std(Ys, axis=1)
    noiselevels = noiselevels * Yvar
    noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
    return Ys + noise

def simulate(method, data, Ys, B, C, Ts, s, true_cell_scores, 
             report_cell_scores = True, QC_phenotypes = False):
    if Ts is None:
        Ts = np.array([None]*len(Ys))
    elif len(Ts.shape) == 2:
        Ts = np.array([Ts]*len(Ys))

    zs, fwers, ntests, beta_vals, beta_pvals, est_cell_scores, others = \
        list(), list(), list(), list(), list(), list(), list()
    interpretabilities = list()

    t0 = time.time()
    
    if QC_phenotypes:
        print(true_cell_scores.index)
        retain_phenos = phenos_uncorr_with_batch(data, true_cell_scores.to_numpy(dtype = "float64").T, 0.3) # Correlation with batch
        true_cell_scores = true_cell_scores.iloc[retain_phenos,:]
        Ys = Ys[retain_phenos,:]
            
    for i, (Y, T) in enumerate(zip(Ys, Ts)):
        print('===', i, ':', time.time() - t0)
        print(Y.shape)

        # run method
        f = getattr(methods, method)
        z, fwer, ntest, beta_val, beta_pval, est_cell_score, other = f(
            data, Y, B, C, T, s)
        zs.append(z)
        fwers.append(fwer)
        ntests.append(ntest)
        beta_vals.append(beta_val)
        beta_pvals.append(beta_pval)
        est_cell_scores.append(est_cell_score)
        others.append(other)
        interpretabilities.append(
            np.corrcoef(true_cell_scores.values[i].astype(np.float), est_cell_score)[0,1])
        print('CORR:', interpretabilities[-1])

        # print update for debugging
        nsig = (fwer <= 0.05).sum()
        print('min fwer:', fwer.min())
        if nsig > 0:
            print('***nsig:', nsig)
        else:
            print('nsig:', nsig)

        gc.collect()
    if report_cell_scores:
            return {'zs':np.array(zs),
            'fwers':np.array(fwers),
            'ntests':np.array(ntests),
            'beta_vals':np.array(beta_vals),
            'beta_pvals':np.array(beta_pvals),
            'est_cell_scores':np.array(est_cell_scores),
            'true_cell_scores':true_cell_scores,
            'others':np.array(others),
            'interpretabilities':np.array(interpretabilities)
            }
    else:
        return {'zs':np.array(zs),
            'fwers':np.array(fwers),
            'ntests':np.array(ntests),
            'beta_vals':np.array(beta_vals),
            'beta_pvals':np.array(beta_pvals),
            'est_cell_scores': None, 
            'true_cell_scores': None, 
            'others':np.array(others),
            'interpretabilities':np.array(interpretabilities)
            }
