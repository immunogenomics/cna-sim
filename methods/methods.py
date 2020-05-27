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


def _expgrowth(*args, **kwargs):
    data, Y, B, C, T, s = args
    kwargs.update({'seed':None})
    z, fwer, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                data.uns['neighbors']['connectivities'], Y, B=B, C=C, T=T, s=s,
                skip_fdr=True,
                **kwargs)
    return z, fwer, ntest, t
def _expgrowthfdr(*args, **kwargs):
    data, Y, B, C, T, s = args
    kwargs.update({'seed':None})
    z, fwer, fdr, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                data.uns['neighbors']['connectivities'], Y, B=B, C=C, T=T, s=s,
                skip_fdr=False,
                **kwargs)
    return z, fdr, ntest, t
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
def expgrowthfdr_avg_nt20_gr5_ms50_Nn100(*args):
    return _expgrowthfdr(*args,
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

