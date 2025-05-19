from math import factorial
import numpy as np


def fac(n):
    
    if n <= 1:
        
        return 1
    
    else:
        
        return factorial(n)
    
def double_fac(n):
    
    if n <= 1:
        
        return 1
    
    else:
        prod = 1
        while n > 1:
            
            prod *= n 
            n+= -2
            
        return prod

def calc_freq(m,k,s):
    
    coeffs = np.zeros(2*k+1)
    coeffs[0] = m * fac(2*(k+m)) / fac(k)/fac(k+m)
    
    for i in range(1,2*k+1):
        
        if i % 2 == 0:
            
            n = i // 2
            
            coeffs[i] = (-1)**(2*k-n)*fac(2*(k+n+m))/fac(k-n)/fac(2*n)/fac(k+n+m)*(2*n+m)
            
        else:
            
            n = (i+1)//2
            
            coeffs[i] = -2 * (-1)**(2*k-n)*fac(2*(k+n+m))/fac(k-n)/fac(2*n)/fac(k+n+m) * n
            
    roots = np.roots(coeffs[::-1])
    
    p = np.argsort(np.abs(roots))
    
    roots = roots[p]

    return roots[s]*2


print(calc_freq(2,3,[0,1,2,3,4,5]))
