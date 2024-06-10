import numpy as np
from sphrflow_main import LN
import scipy.sparse.linalg as spla
from time import time
def iterate_soln(PDE_mat,freq_mat_1,n_it,tol,step,freq0,soln0,rhs,step0=0):
    
    t0 = time()
    freq = freq0
    freqs = []
    steps = []
    k = step0
    while True:
        
        freq += step
        k += 1
        
        
        print(k)
        soln = soln0
        res = np.inf

        for i in range(n_it):
            
            freq_mat = (freq-freq0)*freq_mat_1
            
            soln = LU.solve(rhs-freq_mat @ soln)
            res = np.linalg.norm((PDE_mat+freq_mat) @ soln-rhs)
            
            if res < tol:
                freqs.append(freq)
                steps.append(k)
                np.savetxt('soln_{:d}.txt'.format(k),soln)
                
                break
        
        if res >= tol:
            
            break
    
    print(time()-t0)
    return freqs,steps

    


N = 200
l_max = 60
rad_ratio = 0.0
m = 0

for_freq = 0.939
ek = 1e-4
bc_list = [['tor','t',1,2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]


LN_case = LN(N,rad_ratio,m,l_max,eigen_flag=0)

tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b = LN_case.gen_bc_arrs(bc_list)
rhs_tor_odd,rhs_tor_even = LN_case.gen_rhs(tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b)


max_outer = 2000
k = 86
freq0 = 0.086


n_it = 50
tol = 1e-6
step = 1e-3

freq_mat_1 = LN_case.gen_freq_matrix('tor',1.)
while k < max_outer:
    

    PDE_mat = LN_case.gen_PDE_matrix('tor',freq0,ek)
    LU = spla.splu(PDE_mat)
    soln0 = LU.solve(rhs_tor_odd)
    
    
    freqs,steps = iterate_soln(PDE_mat,freq_mat_1,n_it,tol,step,freq0,soln0,rhs_tor_odd,step0=k)

    print(freqs[-1],steps[-1])
    
    freq0 = freqs[-1]
    k = steps[-1]



    