import numpy as np
from scipy.special import lpmv  
def l_setup(m,l_max):
    
    if m == 0:
        
        l_min = 1
    
    else:
        
        l_min = m
    
    if l_min % 2 == 0:
        
        l_odd = [i for i in range(l_min+1,l_max,2)]
        l_even = [i for i in range(l_min+2,l_max,2)]
    
    else:
        
        l_odd = [i for i in range(l_min+2,l_max,2)]
        l_even = [i for i in range(l_min+1,l_max,2)]
        
    if l_min % 2 == 0:
        
        l_even_full = [l_min] + l_even
        l_odd_full = l_odd
    else:
        
        l_odd_full = [l_min] + l_odd
        l_even_full = l_even
        
    if l_max % 2 == 0:
        
        l_even_full = l_even_full + [l_max]
    
    else:
        
        l_odd_full = l_odd_full + [l_max]
        
    n_l = l_max - l_min + 1
        
    return l_min,n_l,l_odd,l_even,l_even_full,l_odd_full


def trun_fact(l,m):
    
    
    prod = 1

    for k in range(l-m+1,l+m+1):
        
        prod *= k
        
    return prod


def sphrharm(l,m,theta,phi):
    # spherical harmonics degree l order m 
    
    if l < m:
        
        N = 0
    
    if m == 0:
        N = ((2*l+1)/4/np.pi )**(1/2)
    
    elif l >= m:
        N = ((2*l+1)/4/np.pi / trun_fact(l,m))**(1/2)
    
    else:
        N = 0
    
    if m % 2 == 1:
        
        N*=-1 
        # this cancels the Cordon-Shortley 
        # phase factor present by default
        # in scipy assoc. Legendre funcs
    
    return N*lpmv(m,l,np.cos(theta))*np.exp(1j*m*phi)


def Ekman_disp(freq,eps,E,eta):
    
    return 2*np.pi/15/np.sqrt(2) * np.abs(eps)**2 * E**(1/2)/(1-eta)**4 * ((2-freq)**(5/2)+(2+freq)**(5/2)-1/7*((2-freq)**(7/2)+(2+freq)**(7/2)))

 
def Ekman_kin(freq,eps,E,eta):
    
    return np.pi/3/np.sqrt(2) * eps**2*E**(1/2)/(1-eta)**4 * ((2-freq)**(3/2)+(2+freq)**(3/2)-1/5*((2-freq)**(5/2)+(2+freq)**(5/2)))
 