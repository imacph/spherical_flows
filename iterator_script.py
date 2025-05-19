import numpy as np
from sphrflow_main import Iterator
import sys
    



f_idx = 1
N =220
l_max =220
rad_ratio = 0.
m = 0


ek = 10**(-8)

bc_list = [['tor','t',1,2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]

iterator = Iterator(N,rad_ratio,m,l_max)

freq_array = np.arange(0*(f_idx-1),0.2*f_idx,1e-4)


path = '/scratch/ianm02/dissipation_spectrum_recalc/ekman_8_{:d}/power.txt'.format(f_idx)

print(path)
iterator.iterate(freq_array,ek,bc_list,'tor',max_inner_it=50,tol=1e-3,savepath=path)

