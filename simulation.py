import time, gc
import numpy as np
from methods import methods
from argparse import Namespace

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
        pheno_cor_wBatch = np.abs(np.corrcoef(np.hstack((cell_scores[:,i_pheno:(i_pheno+1)], onehot_batch)), rowvar = False)[0,1:-1])
        if np.sum(pheno_cor_wBatch > cor_thresh) ==0:
            final_set.append(i_pheno)
    return final_set

# Returns the labels for clusters that pass checks for required sample representation
# and lack of correlation with batch
def discard_bad_clusters(data, cluster_res, min_cells_per_sample, min_samples_per_cluster,
                         clust_batch_cor_thresh):
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
             report_cell_scores=True, QC_phenotypes=False):
    if Ts is None:
        Ts = np.array([None]*len(Ys))
    elif len(Ts.shape) == 2:
        Ts = np.array([Ts]*len(Ys))

    t0 = time.time()
    if QC_phenotypes:
        print(true_cell_scores.index)
        retain_phenos = phenos_uncorr_with_batch(
            data, true_cell_scores.to_numpy(dtype="float64").T, 0.3) # Correlation with batch
        true_cell_scores = true_cell_scores.iloc[retain_phenos,:]
        Ys = Ys[retain_phenos,:]

    for i, (Y, T, (pheno, truth)) in enumerate(zip(Ys, Ts, true_cell_scores.iterrows())):
        print('===', i, pheno, ':', time.time() - t0)
        print(Y.shape)

        # run method
        f = getattr(methods, method)
        p, cell_scores, cell_sigs, other = f(data, Y, B, C, T, s)

        ix = ~np.isnan(cell_scores)
        interp = \
            np.corrcoef(
                truth.astype(float)[ix], cell_scores[ix]
                )[0,1]

        print('p:', p)
        if p < 0.05:
            print('***sig')
        print('interp:', interp)

        # print memory usage
        import os
        import psutil
        process = psutil.Process(os.getpid())
        print('mem usage:', process.memory_info().rss / 1e6, 'MB')

        gc.collect()
        if report_cell_scores:
            yield Namespace(**{'pheno':pheno,
                        'i':i,
                        'p':p,
                        'truth':truth,
                        'cell_scores':cell_scores,
                        'cell_sigs':cell_sigs,
                        'interp':interp,
                        'other':other})
        else:
            yield Namespace(**{'pheno':pheno,
                        'i':i,
                        'p':p,
                        'interp':interp,
                        'other':other})
