import mcsc as mc

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
    corrs = []

    t0 = time()
    for c in sorted(df.m_cluster.unique().astype(int)):
        print(time()-t0, ': cluster', c, 'of', len(df.m_cluster.unique()))
        df['cluster'] = df.m_cluster == str(c)
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

    p = np.array(ps)
    fwer = p * len(p)
    z = np.sqrt(st.chi2.isf(p, 1))
    return z, fwer, len(z), None
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


import numpy as np
import scipy.stats as st
def _linreg(*args, **kwargs):
    data, Y, B, C, T, s = args
    p, _, _ = mc.tl._pfm.linreg(data, Y, B, T, **kwargs)
    return np.array([np.sqrt(st.chi2.isf(p, 1))]), \
        np.array([p]), \
        1, \
        None
def _minp(*args, **kwargs):
    data, Y, B, C, T, s = args
    _, _, betap = mc.tl._pfm.linreg(data, Y, B, T, **kwargs)
    return np.array([np.sqrt(st.chi2.isf(betap, 1))]), \
        betap * len(betap), \
        len(betap), \
        None

def linreg_nfm_npcs10_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=10, L=0)
def linreg_nfm_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=0)
def linreg_nfm_npcs30_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=30, L=0)
def linreg_nfm_npcs40_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=40, L=0)
def linreg_nfm_npcs50_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=50, L=0)
def linreg_nfm_npcs100_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=100, L=0)

def linreg_nfm_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=0)
def linreg_nfm_npcs20_L1em4(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e-4)
def linreg_nfm_npcs20_L1em2(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e-2)
def linreg_nfm_npcs20_L1e0(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1)
def linreg_nfm_npcs20_L1e2(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e2)
def linreg_nfm_npcs20_L1e4(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e4)
def linreg_nfm_npcs20_L1e6(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e6)
def linreg_nfm_npcs20_L1e8(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e8)
def linreg_nfm_npcs20_L1e10(*args):
    return _linreg(*args, repname='sampleXnh', npcs=20, L=1e10)

def linreg_dleiden0p2_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXdleiden0p2', npcs=20, L=0)
def linreg_dleiden1_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXdleiden1', npcs=20, L=0)
def linreg_dleiden2_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXdleiden2', npcs=20, L=0)
def linreg_dleiden5_npcs20_L0(*args):
    return _linreg(*args, repname='sampleXdleiden5', npcs=20, L=0)


def _mixedmodel(*args, **kwargs):
    data, Y, B, C, T, s = args
    p, _, _ = mc.tl._pfm.mixedmodel(data, Y, B, T, **kwargs)
    return np.array([np.sqrt(st.chi2.isf(p, 1))]), \
        np.array([p]), \
        1, \
        None

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






