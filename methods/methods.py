import cna
import numpy as np
import scipy.stats as st

def _MASC(data, Y, B, C, T, s, clustertype):
    import pandas as pd
    import numpy as np
    import scipy.stats as st
    from time import time
    import os, tempfile

    # prepare data
    df = data.obs[['id',clustertype]].rename(columns={clustertype:'m_cluster'})
    df['batch'] = np.repeat(B, C)
    df['phenotype'] = np.repeat(Y, C)
    othercols = []
    if s is not None:
        print('coding cellular covariates')
        for i, s_ in enumerate(s.T):
            bins = np.linspace(np.min(s_), np.max(s_)+1e-7, 6)
            df['s'+str(i)] = np.digitize(s_, bins)
            othercols.append('s'+str(i))
    if T is not None:
        for i, T_ in enumerate(T.T):
            df['T'+str(i)] = np.repeat(T_, C)
            othercols.append('T'+str(i))
    ps = []
    betas = []

    t0 = time()
    for c in sorted(df.m_cluster.unique().astype(int)):
        print(time()-t0, ': cluster', c, 'of', len(df.m_cluster.unique()))
        df['cluster'] = df.m_cluster.astype(str) == str(c)
        print(df.cluster.sum(), 'cells in cluster', c)
        if np.sum(df['cluster'])==1:
            continue
        df_w = df.groupby(['id', 'batch', 'phenotype', 'cluster']+othercols, observed=True
                        ).size().to_frame(name='weight').reset_index()
        df_t = df.id.value_counts()
        print(df_w.shape)

        temp = tempfile.NamedTemporaryFile(mode='w+t')
        df_w.to_csv(temp, sep='\t', index=False)
        temp.flush()

        #execute MASC
        mascscript = os.path.dirname(__file__) + '/runmasc.R'
        command = 'Rscript '+ mascscript + ' ' + temp.name + ' ' +' '.join(othercols)
        stream = os.popen(command)
        for line in stream:
            if line == '***RESULTS\n':
                break
        result = pd.read_csv(stream, delim_whitespace=True)
        temp.close()

        # process results and return
        print(result)

        y = df_w[df_w.cluster].set_index('id').weight / df_t

        ps.append(result['model.pvalue'].values[0])
        betas.append(-result['model.beta'].values[0]) # NOTE SIGN FLIP -- because runmasc
                                                    # reports beta for the cells not in the
                                                    # cluster

    cell_scores = np.zeros(len(data))
    cell_sigs = np.zeros(len(data))
    for c, p, beta in zip(sorted(df.m_cluster.unique().astype(int)), ps, betas):
        cell_scores[df.m_cluster.astype(int) == c] = beta
        cell_sigs[df.m_cluster.astype(int) == c] = p
    ps = np.array(ps)
    betas = np.array(betas)

    return np.min(ps)*len(ps), cell_scores, cell_sigs < 0.05/len(ps), (betas, ps)
def MASC_leiden0p2(*args):
    return _MASC(*args, clustertype='leiden0p2')
def MASC_leiden1(*args):
    return _MASC(*args, clustertype='leiden1')
def MASC_leiden2(*args):
    return _MASC(*args, clustertype='leiden2')
def MASC_leiden5(*args):
    return _MASC(*args, clustertype='leiden5')

# return p, cell_scores, cell_significance, (betas, ps)
########################################
def CNA(*args, **kwargs):
    data, Y, B, C, T, s = args
    if 'suffix' in kwargs:
        suffix = kwargs['suffix']
    else:
        suffix = ''

    res = cna.tl.association(data, Y, batches=B, covs=T,  **kwargs)
    data.obs.loc[res.kept, 'ncorrs'] = res.ncorrs
    data.obs.loc[~res.kept, 'ncorrs'] = np.nan
    cell_scores = data.obs.ncorrs.copy().values

    if res.fdr_5p_t is None:
        res.fdr_5p_t = np.inf
    print('fdr threshold is', res.fdr_5p_t)

    return res.p, \
        cell_scores, \
        np.abs(cell_scores) > res.fdr_5p_t, \
        (res.fdr_5p_t, res.fdr_10p_t)

def CNAfast(*args, **kwargs):
    return CNA(*args, local_test=False, **kwargs)

def CNAfast_detailed(*args, **kwargs):
    return CNA(*args, local_test=False, Nnull=10000, **kwargs)

def CNAfast_permute_all(*args, **kwargs):
    return CNA(*args, local_test=False, force_permute_all=True, **kwargs)

# return p, cell_scores, cell_significance, (betas, ps)                                                     
########################################   
import meld

def meld_compare(*args, **kwargs):
    data, Y, B, C, T, s = args
    if 'suffix' in kwargs:
        suffix = kwargs['suffix']
    else:
        suffix = ''

    print(Y)
    meld_op = meld.MELD()
    sample_densities = meld_op.fit_transform(data.X, # this is pca for harmonized CCA data  
                                             sample_labels=data.obs['donor'].values)
    sample_densities = sample_densities.values
    sample_likelihoods = sample_densities/sample_densities.sum(axis=1)[:,None]
    cell_scores = sample_likelihoods[:,Y].mean(axis=1)
    return 1, cell_scores, np.repeat(1, len(cell_scores)), (None, None)
