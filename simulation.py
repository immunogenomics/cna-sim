import time, gc
import numpy as np
from methods import methods

def simulate(method, data, Ys, B, C, Ts, s):
    if Ts is None:
        Ts = np.array([None]*len(Ys))
    elif len(Ts.shape) == 2:
        Ts = np.array([Ts]*len(Ys))

    zs, fwers, ntests, others = list(), list(), list(), list()
    t0 = time.time()
    for i, (Y, T) in enumerate(zip(Ys, Ts)):
        print('===', i, ':', time.time() - t0)
        print(Y.shape)

        # run method
        f = getattr(methods, method)
        z, fwer, ntest, other = f(
            data, Y, B, C, T, s)
        zs.append(z)
        fwers.append(fwer)
        ntests.append(ntest)
        others.append(other)

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
            'others':np.array(others)}
