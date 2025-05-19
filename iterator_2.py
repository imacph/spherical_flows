import numpy as np
from scipy import sparse as sp
from time import time
from sphrflow_main import Iterator


    


N =220
l_max =220
rad_ratio = 0.
m = 0


ek = 10**(-8)

bc_list = [['tor','t',1,2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]

iterator = Iterator(N,rad_ratio,m,l_max)

freq_array = np.arange(0,2,1e-5)

iterator.iterate(freq_array,ek,bc_list,'tor',max_inner_it=50,tol=1e-3)

