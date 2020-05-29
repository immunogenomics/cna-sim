import mcsc as mc

def _MASC_original(data, Y, B, C, T, s, clustertype):
    import pandas as pd
    import numpy as np
    import scipy.stats as st
    import os, tempfile

    # prepare data
    df = data.obs[['id',clustertype]].rename(columns={clustertype:'cluster'})
    df['batch'] = np.repeat(B, C)
    df['phenotype'] = np.repeat(Y, C)
    othercols = []
    if s is not None:
        for i, s_ in enumerate(s.T):
            bins = np.linspace(np.min(s_), np.max(s_)+1e-7, 4)
            df['s'+str(i)] = np.digitize(s_, bins)
            othercols.append('s'+str(i))
    if T is not None:
        for i, T_ in enumerate(T.T):
            df['T'+str(i)] = np.repeat(T_, C)
            othercols.append('T'+str(i))
    df = df.groupby(['id', 'batch', 'phenotype', 'cluster']+othercols, observed=True
                    ).size().to_frame(name='weight').reset_index()

    temp = tempfile.NamedTemporaryFile(mode='w+t')
    df.to_csv(temp, sep='\t', index=False)
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
    result.cluster = [
        int(x.split('cluster')[1])
        for x in result.cluster
        ]
    result = result.sort_values(by='cluster')
    print(result)

    p = result['model.pvalue'].values
    fwer = p * len(p)
    z = np.sqrt(st.chi2.isf(p, 1))
    return z, fwer, len(z), None

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
def MASC_louvain(*args):
    return _MASC(*args, clustertype='louvain')
def MASC_leiden0p2(*args):
    return _MASC(*args, clustertype='leiden0p2')
def MASC_leiden0p5(*args):
    return _MASC(*args, clustertype='leiden0p5')
def MASC_leiden1(*args):
    return _MASC(*args, clustertype='leiden1')
def MASC_leiden2(*args):
    return _MASC(*args, clustertype='leiden2')
def MASC_leiden5(*args):
    return _MASC(*args, clustertype='leiden5')
def MASC_leiden10(*args):
    return _MASC(*args, clustertype='leiden10')
def MASC_dleiden0p2(*args):
    return _MASC(*args, clustertype='dleiden0p2')
def MASC_dleiden0p5(*args):
    return _MASC(*args, clustertype='dleiden0p5')
def MASC_dleiden1(*args):
    return _MASC(*args, clustertype='dleiden1')
def MASC_dleiden2(*args):
    return _MASC(*args, clustertype='dleiden2')
def MASC_dleiden5(*args):
    return _MASC(*args, clustertype='dleiden5')
def MASC_dleiden10(*args):
    return _MASC(*args, clustertype='dleiden10')


import numpy as np
import scipy.stats as st
def _linreg(*args, **kwargs):
    data, Y, B, C, T, s = args
    p = mc.tl._newdiff.linreg(data, Y, B, T, **kwargs)
    return np.array([np.sqrt(st.chi2.isf(p, 1))]), \
        np.array([p]), \
        1, \
        None
def _minp(*args, **kwargs):
    data, Y, B, C, T, s = args
    p = mc.tl._newdiff.marg_minp(data, Y, B, T, **kwargs)
    return np.array([np.sqrt(st.chi2.isf(p, 1))]), \
        np.array([p]), \
        1, \
        None
def _clusterreg(clustername, *args):
    data, Y, B, C, T, s = args
    cols = [clustername + '_'+str(i) for i in range(len(data.obs[clustername].unique()))]
    data.uns['sampleX'+clustername] = data.uns['sampleXmeta'][cols].values
    return _linreg(*args, repname='sampleX'+clustername, nfeatures=None)
def _clusterminp(clustername, *args):
    data, Y, B, C, T, s = args
    cols = [clustername + '_'+str(i) for i in range(len(data.obs[clustername].unique()))]
    data.uns['sampleX'+clustername] = data.uns['sampleXmeta'][cols].values
    return _minp(*args, repname='sampleX'+clustername, nfeatures=None)
def _clusterpcreg_pcs20(clustername, *args):
    data, Y, B, C, T, s = args
    cols = [clustername + '_'+str(i) for i in range(len(data.obs[clustername].unique()))]
    data.uns['sampleX'+clustername] = data.uns['sampleXmeta'][cols].values
    mc.tl._newdiff.pca(data, repname='sampleX'+clustername)
    nfeatures = min(len(cols), 20)
    return _linreg(*args, repname='sampleX'+clustername+'_sampleXpc', nfeatures=nfeatures)

def nnpcreg_ms3_pcs20(*args):
    return _linreg(*args, repname='sampleXnh_sampleXpc', nfeatures=20)

def clusterreg_dleiden0p2(*args):
    return _clusterreg('dleiden0p2', *args)
def clusterreg_dleiden1(*args):
    return _clusterreg('dleiden1', *args)
def clusterreg_dleiden2(*args):
    return _clusterreg('dleiden2', *args)
def clusterreg_dleiden5(*args):
    return _clusterreg('dleiden5', *args)

def clusterminp_dleiden0p2(*args):
    return _clusterminp('dleiden0p2', *args)
def clusterminp_dleiden1(*args):
    return _clusterminp('dleiden1', *args)
def clusterminp_dleiden2(*args):
    return _clusterminp('dleiden2', *args)
def clusterminp_dleiden5(*args):
    return _clusterminp('dleiden5', *args)

def clusterpcreg_pcs20_dleiden0p2(*args):
    return _clusterpcreg_pcs20('dleiden0p2', *args)
def clusterpcreg_pcs20_dleiden1(*args):
    return _clusterpcreg_pcs20('dleiden1', *args)
def clusterpcreg_pcs20_dleiden2(*args):
    return _clusterpcreg_pcs20('dleiden2', *args)
def clusterpcreg_pcs20_dleiden5(*args):
    return _clusterpcreg_pcs20('dleiden5', *args)
















def _expgrowth(*args, **kwargs):
    data, Y, B, C, T, s = args
    kwargs.update({'seed':None})
    z, fwer, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                data.uns['neighbors']['connectivities'], Y, B=B, C=C, T=T, s=s,
                skip_fdr=True,
                **kwargs)
    return z, fwer, ntest, t
def expgrowth_avg_nt20_gr5_ms20(*args):
    return _expgrowth(*args,
        maxsteps=20, nontrivial=20, growthreq=0.05, diffusion=False)
def expgrowth_diff_nt20_gr5_ms20(*args):
    return _expgrowth(*args,
        maxsteps=20, nontrivial=20, growthreq=0.05, diffusion=True)
def expgrowth_avg_nt20_gr5_ms20_Nn500(*args):
    return _expgrowth(*args,
        maxsteps=20, nontrivial=20, growthreq=0.05, diffusion=False, Nnull=500)
def expgrowth_diff_nt20_gr5_ms20_Nn500(*args):
    return _expgrowth(*args,
        maxsteps=20, nontrivial=20, growthreq=0.05, diffusion=True, Nnull=500)
def expgrowth_diff_nt20_gr5_ms50_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=50, nontrivial=20, growthreq=0.05, diffusion=True, Nnull=100)
def expgrowth_avg_nt20_gr5_ms50_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=50, nontrivial=20, growthreq=0.05, diffusion=False, Nnull=100)
def expgrowth_avg_nt100_gr5_ms50_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=50, nontrivial=100, growthreq=0.05, diffusion=False, Nnull=100)
def expgrowth_avg_ntinf_ms50_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=50, nontrivial=100, growthreq=None, diffusion=False, Nnull=100)
def expgrowth_avg_ntinf_ms20_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=20, nontrivial=100, growthreq=None, diffusion=False, Nnull=100)
def expgrowth_avg_ntinf_ms10_Nn100(*args):
    return _expgrowth(*args,
        maxsteps=10, nontrivial=100, growthreq=None, diffusion=False, Nnull=100)
def expgrowth_diff_nt100_gr5_ms50(*args):
    return _expgrowth(*args,
        maxsteps=50, nontrivial=100, growthreq=0.05, diffusion=True)

#######################3
# defunct methods
def _nnreg(*args, **kwargs):
    data, Y, B, C, T, s = args
    kwargs.update({'seed':None})
    z, fwer, _ = \
            mc.tl._newdiff.nnreg(
                data.uns['neighbors']['connectivities'], Y, C, B=B, T=T, s=s,
                **kwargs)
    return z, fwer, None, None
def nnreg_ms10_Nn100(*args):
    return _nnreg(*args,
        maxsteps=10)

def _minfwer(*args, **kwargs):
    data, Y, B, C, T, s = args
    kwargs.update({'seed':None})
    z, fwer, ntest, t, _ = \
            mc.tl._diffusion.diffusion_minfwer(
                data.uns['neighbors']['connectivities'], Y, B=B, C=C, T=T, s=s,
                **kwargs)
    return z, fwer, ntest, t
def minfwer_avg_ms20(*args):
    return _minfwer(*args,
        maxsteps=50, diffusion=False)
def minfwer_diff_ms20(*args):
    return _minfwer(*args,
        maxsteps=50, diffusion=True)

