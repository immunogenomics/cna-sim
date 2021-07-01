data = '/data/srlab1/laurie-yakir/'

tbru = data + 'tbru.real/'
tbru_h5ad = tbru + '2.munged/'

sepsis = data + 'sepsis/'
sepsis_h5ad = sepsis + 'data/'

sim = data + 'sim/'
simdata = sim + 'datasets/'

def simresults(dsetname, simname):
    return sim+'/'+dsetname+'/'+simname+'/'
