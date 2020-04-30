import mcsc as mc

# exp growth stopping criterion, with averaging, for max of 20 steps
# where non-trivial signal is 20 cells, and exponential growth requirement is 5%
def expgrowth_avg_nt20_gr5_ms20(a, Y, B, C, T, s):
    z, fwer, _, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=20,
                    nontrivial=20,
                    growthreq=0.05,
                    diffusion=False,
                    seed=None)
    return z, fwer, ntest

# exp growth stopping criterion, with diffusion, for max of 20 steps
# where non-trivial signal is 20 cells, and exponential growth requirement is 5%
def expgrowth_diff_nt20_gr5_ms20(a, Y, B, C, T, s):
    z, fwer, _, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=20,
                    nontrivial=20,
                    growthreq=0.05,
                    diffusion=True,
                    seed=None)
    return z, fwer, ntest

# exp growth stopping criterion, with averaging, for max of 50 steps
# where non-trivial signal is 100 cells, and exponential growth requirement is 5%
def expgrowth_avg_nt100_gr5_ms50(a, Y, B, C, T, s):
    z, fwer, _, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=50,
                    nontrivial=100,
                    growthreq=0.05,
                    diffusion=False,
                    seed=None)
    return z, fwer, ntest

# exp growth stopping criterion, with diffusion, for max of 50 steps
# where non-trivial signal is 100 cells, and exponential growth requirement is 5%
def expgrowth_diff_nt100_gr5_ms50(a, Y, B, C, T, s):
    z, fwer, _, _, ntest, t = \
            mc.tl._diffusion.diffusion_expgrowth(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=50,
                    nontrivial=100,
                    growthreq=0.05,
                    diffusion=True,
                    seed=None)
    return z, fwer, ntest

# min-FWER stopping criterion, with averaging rather than diffusion, for max of 20 steps
def minfwer_avg_ms20(a, Y, B, C, T, s):
    z, fwer, ntest, t, Nt_f = \
            mc.tl._diffusion.diffusion_minfwer(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=20,
                    diffusion=False,
                    seed=None)
    return z, fwer, ntest

# min-FWER stopping criterion, with diffusion rather than averaging, for max of 20 steps
def minfwer_diff_ms20(a, Y, B, C, T, s):
    z, fwer, ntest, t, Nt_f = \
            mc.tl._diffusion.diffusion_minfwer(
                    a, Y, B=B, C=C, T=T, s=s,
                    maxsteps=20,
                    diffusion=True,
                    seed=None)
    return z, fwer, ntest

