import mcsc as mc
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
        df['cluster'] = df.m_cluster == str(c)
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
        command = 'Rscript /data/srlab1/yakir/mcsc-sim/methods/runmasc.R ' + temp.name + ' ' + \
            ' '.join(othercols)
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
    for c, p, beta in zip(sorted(df.m_cluster.unique().astype(int)), ps, betas):
        if p * len(ps) <= 0.05:
            cell_scores[df.m_cluster.astype(int) == c] = beta

    ps = np.array(ps)
    fwers = ps * len(ps)
    zs = np.sqrt(st.chi2.isf(ps, 1))
    betas = np.array(betas)
    return zs, fwers, len(zs), betas, ps, cell_scores, None
def MASC_leiden0p2(*args):
    return _MASC(*args, clustertype='leiden0p2')
def MASC_leiden1(*args):
    return _MASC(*args, clustertype='leiden1')
def MASC_leiden2(*args):
    return _MASC(*args, clustertype='leiden2')
def MASC_leiden5(*args):
    return _MASC(*args, clustertype='leiden5')
def MASC_dleiden0p2(*args):
    return _MASC(*args, clustertype='dleiden0p2')
def MASC_dleiden1(*args):
    return _MASC(*args, clustertype='dleiden1')
def MASC_dleiden2(*args):
    return _MASC(*args, clustertype='dleiden2')
def MASC_dleiden5(*args):
    return _MASC(*args, clustertype='dleiden5')

########################################
def cnav3(*args, **kwargs):
    data, Y, B, C, T, s = args
    if 'suffix' in kwargs:
        suffix = kwargs['suffix']
    else:
        suffix = ''

    res = mc.tl._pfm.association(data, Y, B, T, **kwargs)

    data.obs.loc[data.uns['keptcells'+suffix], 'ncorrs'] = res.ncorrs
    data.obs.loc[~data.uns['keptcells'+suffix], 'ncorrs'] = np.nan
    #TODO: set cells with FDR>5% to have estimated cell scores of 0 versus nan?

    return np.array([np.sqrt(st.chi2.isf(res.p, 1))]), \
        np.array([res.p]), \
        1, \
        None, \
        None, \
        data.obs.ncorrs.values, \
        None

def cnav3_3steps(*args, **kwargs):
    return cnav3(*args, **kwargs, suffix='_3steps')

########################################
def _mixedmodel(*args, **kwargs):
    def diffuse_phenotype(data, s, nsteps=3):
        a = data.uns['neighbors']['connectivities']
        colsums = np.array(a.sum(axis=0)).flatten() + 1

        for i in range(nsteps):
            s = a.dot(s/colsums) + s/colsums
        return s

    data, Y, B, C, T, s = args
    res = mc.tl._pfm.mixedmodel(data, Y, B, T, **kwargs)
    cell_scores = diffuse_phenotype(data, res.beta)

    return np.array([np.sqrt(st.chi2.isf(res.p, 1))]), \
        np.array([res.p]), \
        1, \
        res.gamma, \
        res.gamma_p, \
        res.beta, \
        cell_scores

def mixedmodel_nfm_npcs10(*args):
    return _mixedmodel(*args, repname='sampleXnh', npcs=10)
def mixedmodel_nfm_npcs20(*args):
    return _mixedmodel(*args, repname='sampleXnh', npcs=20)
def mixedmodel_nfm_npcs30(*args):
    return _mixedmodel(*args, repname='sampleXnh', npcs=30)
def mixedmodel_nfm_npcs40(*args):
    return _mixedmodel(*args, repname='sampleXnh', npcs=40)
def mixedmodel_nfm_npcs50(*args):
    return _mixedmodel(*args, repname='sampleXnh', npcs=50)

def mixedmodel_cfm_leiden2(*args):
    return _mixedmodel(*args, repname='sampleXleiden2', npcs=20)
def mixedmodel_cfm_leiden5(*args):
    return _mixedmodel(*args, repname='sampleXleiden5', npcs=20)
def mixedmodel_cfm_leiden10(*args):
    return _mixedmodel(*args, repname='sampleXleiden10', npcs=20)
