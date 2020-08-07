import time, gc
import numpy as np
from methods import methods

def avg_within_sample(data, true_cell_scores):
    cols = true_cell_scores.columns.values
    true_cell_scores['id'] = data.obs.id

    # this is to use the fact that pandas will match appropriately by sample
    data.uns['sampleXmeta'][cols] = true_cell_scores.groupby('id').aggregate(np.mean)[cols]
    Ys = data.uns['sampleXmeta'][cols]
    data.uns['sampleXmeta'].drop(columns=cols, inplace=True)
    return Ys.T

def add_noise(Ys, noiselevels): #Ys is assumed to have one row per phenotype
    Yvar = np.std(Ys, axis=1)
    noiselevels = noiselevels * Yvar
    noise = np.random.randn(*Ys.shape) * noiselevels[:,None]
    return Ys + noise

def simulate(method, data, Ys, B, C, Ts, s, true_cell_scores):
    if Ts is None:
        Ts = np.array([None]*len(Ys))
    elif len(Ts.shape) == 2:
        Ts = np.array([Ts]*len(Ys))

    zs, fwers, ntests, beta_vals, beta_pvals, est_cell_scores, others = \
        list(), list(), list(), list(), list(), list(), list()
    interpretabilities = list()

    t0 = time.time()
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
