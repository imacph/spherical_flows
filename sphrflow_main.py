import numpy as np
from scipy import sparse as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import lpmv
import scipy.sparse.linalg as spla
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


class Iterator:
    
    def __init__(self,N,rad_ratio,m,l_max):
        
        self.matrix_builder = Matrix_builder_forced(N,rad_ratio,m,l_max)
        self.rhs_builder = Rhs_builder(N,rad_ratio,m,l_max)

    def iterate(self,freq_array,ek,bc_list,odd_flag,tol=1e-4,max_inner_it=30):
        
        self.rhs_builder.gen_rhs(bc_list)
        
        base_freq_mat = self.matrix_builder.gen_freq_matrix(odd_flag,1.)
        self.rhs_builder.gen_rhs(bc_list,odd_flag)
        
        
        n_freq = len(freq_array)
        k = 0 
        while k <= n_freq-1:
            
            freq0 = freq_array[k]
            PDE_mat = self.matrix_builder.gen_PDE_matrix(odd_flag,freq_array[k],ek)
            
            LU = spla.splu(PDE_mat)
            soln0 = LU.solve(self.rhs_builder.rhs)
            
            
            inner_iterating = True
            n_suc_it = 0
            while inner_iterating:
                
                inner_k = 0
                res = np.inf
                soln = soln0
                while inner_k < max_inner_it-1 and res > tol:
                    
                    freq = freq_array[k]
                    
                    freq_mat = (freq-freq0)*base_freq_mat
                    
                    soln = LU.solve(self.rhs_builder.rhs-freq_mat @ soln)
                    res = np.linalg.norm((PDE_mat+freq_mat) @ soln-self.rhs_builder.rhs)
                    
                    inner_k += 1
            
                if inner_k == max_inner_it-1 and res > tol:
                    
                    inner_iterating = False
                    
                else:
                    
                    n_suc_it += 1
                    
            k += 1 + n_suc_it
                    
                    
            

        
        
class Matrix_builder_forced:
    
    def __init__(self,N,rad_ratio,m,l_max):
        
        self.N = N
        self.rad_ratio = rad_ratio
        self.m = m
        self.l_max = l_max
        
        self.r_end = 1/(1-rad_ratio)
        self.r_start = self.r_end * rad_ratio
        
        self.r_fac = 2/(self.r_end-self.r_start)
        
        
        if self.rad_ratio > 0.0:
            self.x_grid = np.cos(np.linspace(0,N,N+1)*np.pi/N)
            self.r_grid = self.x_grid * (self.r_end-self.r_start)/2 + (self.r_end+self.r_start)/2
            self.eval_mat,self.df1_mat,self.df2_mat,self.df4_mat = self.gen_deriv_mats()
        
        elif self.rad_ratio == 0.0:
            self.x_grid = np.cos(np.linspace(0,N,N+1)*np.pi/(2*N+1))
            self.r_grid = self.x_grid
            self.eval_mat_even,self.eval_mat_odd,self.df1_mat_even,self.df1_mat_odd,self.df2_mat_even,self.df2_mat_odd,self.df4_mat_even,self.df4_mat_odd = self.gen_deriv_mats()
        
        self.l_min,self.n_l,self.l_odd,self.l_even,self.l_even_full,self.l_odd_full = l_setup(self.m,self.l_max)
        
        
        self.orr_mat = sp.csr_matrix(np.diag(1/self.r_grid),dtype=complex)
        self.orr_sqr_mat = self.orr_mat * self.orr_mat
            
    
        self.c_l = np.zeros(self.n_l+1)
        
        for l in range(self.l_min,self.l_max+2):
            
            i = l - self.l_min
            
            self.c_l[i] = np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))
        
    
            

    def gen_deriv_mats(self):
        
        
        if self.rad_ratio > 0.0:
            df1_mat = sp.csr_matrix((self.N+1,self.N+1))
            df2_mat = sp.csr_matrix((self.N+1,self.N+1))
            df4_mat = sp.csr_matrix((self.N+1,self.N+1))
            eval_mat = sp.csr_matrix((self.N+1,self.N+1))
            
            for n in range(self.N+1):
                
                eval_mat[:,n] = np.cos(n*np.arccos(self.x_grid))
                
                df1_mat[0,n] = n*n
                df1_mat[-1,n] = (-1)**(n+1)*n*n
                
                df2_mat[0,n] = n*n * (n*n-1)/3
                df2_mat[-1,n] =(-1)**(n+2)* df2_mat[0,n]
                
                df4_mat[0,n] = n*n * (n*n-1)/3 * (n*n-4)/5 * (n*n-9)/7
                df4_mat[-1,n] = (-1)**(n+2)* df4_mat[0,n]
                
                
                df1_mat[1:-1,n] = n * np.sin(n*np.arccos(self.x_grid[1:-1]))/np.sin(np.arccos(self.x_grid[1:-1]))
                df2_mat[1:-1,n] = n*self.x_grid[1:-1]*np.sin(n*np.arccos(self.x_grid[1:-1]))/(1-self.x_grid[1:-1]**2)**(3/2) - n*n * np.cos(n*np.arccos(self.x_grid[1:-1]))/(1-self.x_grid[1:-1]**2)
                df4_mat[1:-1,n] = n * ((n*(n**2*(self.x_grid[1:-1]**2-1)+11*self.x_grid[1:-1]**2+4)*np.cos(n*np.arccos(self.x_grid[1:-1])))/(self.x_grid[1:-1]**2-1)**3+(3*self.x_grid[1:-1]*(2*n**2*(self.x_grid[1:-1]**2-1)+2*self.x_grid[1:-1]**2+3)*np.sin(n*np.arccos(self.x_grid[1:-1])))/(1-self.x_grid[1:-1]**2)**(7/2))
                
                
                
            return eval_mat,self.r_fac*df1_mat,self.r_fac**2*df2_mat,self.r_fac**4*df4_mat
        
        elif self.rad_ratio == 0.0:
            df1_mat = sp.csr_matrix((self.N+1,self.N+1))
            df2_mat = sp.csr_matrix((self.N+1,self.N+1))
            df3_mat = sp.csr_matrix((self.N+1,self.N+1))
            df4_mat = sp.csr_matrix((self.N+1,self.N+1))
            eval_mat = sp.csr_matrix((self.N+1,self.N+1))
            
            df1_mat_odd = sp.csr_matrix((self.N+1,self.N+1))
            df2_mat_odd = sp.csr_matrix((self.N+1,self.N+1))
            df4_mat_odd = sp.csr_matrix((self.N+1,self.N+1))
            eval_mat_odd = sp.csr_matrix((self.N+1,self.N+1))
            
            df1_mat_even = sp.csr_matrix((self.N+1,self.N+1))
            df2_mat_even = sp.csr_matrix((self.N+1,self.N+1))
            df4_mat_even = sp.csr_matrix((self.N+1,self.N+1))
            eval_mat_even = sp.csr_matrix((self.N+1,self.N+1))
            
            grid_diag = sp.csr_matrix(np.diag(self.x_grid),dtype=complex)
            grid_sqr_diag = grid_diag * grid_diag
            
            for i in range(self.N+1):

                
                
                n = 2*i+1
                
                eval_mat[:,i] = np.cos(n*np.arccos(self.x_grid))
                
                df1_mat[0,i] = n*n
                df2_mat[0,i] = n*n * (n*n-1)/3
                df3_mat[0,i] = n*n * (n*n-1)/3 * (n*n-4)/5
                df4_mat[0,i] = n*n * (n*n-1)/3 * (n*n-4)/5 * (n*n-9)/7

                df1_mat[1:,i] = n * np.sin(n*np.arccos(self.x_grid[1:]))/np.sin(np.arccos(self.x_grid[1:]))
                df2_mat[1:,i] = n*self.x_grid[1:]*np.sin(n*np.arccos(self.x_grid[1:]))/(1-self.x_grid[1:]**2)**(3/2) - n*n * np.cos(n*np.arccos(self.x_grid[1:]))/(1-self.x_grid[1:]**2)
                df3_mat[1:,i] = n/(1-self.x_grid[1:]**2)**(5/2) * ((n**2*(self.x_grid[1:]**2-1)+2*self.x_grid[1:]**2+1)*np.sin(n*np.arccos(self.x_grid[1:]))-3*n*self.x_grid[1:]*np.sqrt(1-self.x_grid[1:]**2)*np.cos(n*np.arccos(self.x_grid[1:])))
                
                df4_mat[1:,i] = n * ((n*(n**2*(self.x_grid[1:]**2-1)+11*self.x_grid[1:]**2+4)*np.cos(n*np.arccos(self.x_grid[1:])))/(self.x_grid[1:]**2-1)**3+(3*self.x_grid[1:]*(2*n**2*(self.x_grid[1:]**2-1)+2*self.x_grid[1:]**2+3)*np.sin(n*np.arccos(self.x_grid[1:])))/(1-self.x_grid[1:]**2)**(7/2))
                
                
                eval_mat_odd[:,i] = grid_sqr_diag*eval_mat[:,i]
                
                df1_mat_odd[:,i] = grid_sqr_diag * df1_mat[:,i]  + 2*grid_diag*eval_mat[:,i]

                df2_mat_odd[:,i] = grid_sqr_diag * df2_mat[:,i]   + 4*grid_diag*df1_mat[:,i] + 2 * eval_mat[:,i]
                
                df4_mat_odd[:,i] = grid_sqr_diag  * df4_mat[:,i] + 8 * grid_diag * df3_mat[:,i] + 12 * df2_mat[:,i]
                
                
                eval_mat_even[:,i] = grid_diag*eval_mat[:,i]
                
                df1_mat_even[:,i] = grid_diag*df1_mat[:,i] + eval_mat[:,i]
                
                df2_mat_even[:,i] = grid_diag*df2_mat[:,i] + 2*df1_mat[:,i]
                
                df4_mat_even[:,i] = grid_diag*df4_mat[:,i]  + 4 *df3_mat[:,i]
                
                
            return eval_mat_even,eval_mat_odd,df1_mat_even,df1_mat_odd,df2_mat_even,df2_mat_odd,df4_mat_even,df4_mat_odd
    
    


    def gen_block_diag_freq(self,field,l,for_freq):
        
        lp = l*(l+1)
        lp2 = lp*lp
        
        diag_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        
        if self.rad_ratio > 0.0:
        
            if field == 'toroidal':
                
                diag_mat[1:-1,:] += (1j *lp*for_freq*self.eval_mat)[1:-1,:]
            
                
            elif field == 'poloidal':
                
                diag_mat[2:-2,:] = 1j * lp2*for_freq  * (self.orr_sqr_mat*self.eval_mat)[2:-2,:]

                diag_mat[2:-2,:] +=-1j * lp*for_freq* self.df2_mat[2:-2,:]

        
        elif self.rad_ratio == 0.0:
            
            if field == 'toroidal' and l % 2 == 0:
                
                diag_mat[1:] += (1j * lp*for_freq * self.eval_mat_odd)[1:]


            elif field == 'toroidal':
                
                diag_mat[1:] += (1j * lp*for_freq * self.eval_mat_even)[1:]


                
                
            elif field == 'poloidal' and l % 2 == 0:
                
                diag_mat[2:] = 1j * lp2*for_freq * (self.orr_sqr_mat*self.eval_mat_odd)[2:]

                diag_mat[2:] +=-1j * lp*for_freq* self.df2_mat_odd[2:]

                


            elif field == 'poloidal':
                
                diag_mat[2:] = 1j * lp2*for_freq * (self.orr_sqr_mat*self.eval_mat_even)[2:]

                diag_mat[2:] +=-1j * lp*for_freq* self.df2_mat_even[2:]


        return diag_mat
        
    
    def gen_block_row_spatial(self,field,l,ek):
        
        lp = l*(l+1)
        lp2 = lp*lp
        
        lower_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        upper_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        diag_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        
        if self.rad_ratio > 0.0:
        
            if field == 'toroidal':
                
                diag_mat[1:-1,:] += (1j * (-2*self.m) * self.eval_mat + ek*lp2*self.orr_sqr_mat*self.eval_mat)[1:-1,:]
                diag_mat[1:-1,:] += -lp * ek * self.df2_mat[1:-1,:]
                
                diag_mat[0,:] = self.eval_mat[0,:]
                #diag_mat[-1,:] = self.eval_mat[-1,:]
                diag_mat[-1,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[-1,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:-1] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[1:-1]
                upper_mat[1:-1] += -upper_fac * self.df1_mat[1:-1]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:-1] += lower_fac * l*(self.orr_mat*self.eval_mat)[1:-1]
                lower_mat[1:-1] += -lower_fac * self.df1_mat[1:-1]
                
                
            elif field == 'poloidal':
                
                diag_mat[2:-2,:] = 1j * (-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat)[2:-2,:]
                diag_mat[2:-2,:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat)[2:-2,:]
                diag_mat[2:-2,:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat)[2:-2,:]
                diag_mat[2:-2,:] +=-1j * (-2*self.m)* self.df2_mat[2:-2,:]
                diag_mat[2:-2,:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat)[2:-2,:]
                diag_mat[2:-2,:] += lp * ek * self.df4_mat[2:-2,:]
                
                diag_mat[0,:] = self.eval_mat[0,:]
                diag_mat[-1,:] = self.eval_mat[-1,:]
                diag_mat[1,:] = self.df1_mat[0,:]
                #diag_mat[-2,:] = self.df1_mat[-1,:]
                diag_mat[-2,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[-1,:]
                
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:-2,:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[2:-2,:]
                upper_mat[2:-2,:] += -upper_fac * self.df1_mat[2:-2,:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[2:-2,:] += lower_fac * l*(self.orr_mat*self.eval_mat)[2:-2,:]
                lower_mat[2:-2,:] += -lower_fac * self.df1_mat[2:-2,:]
                
            
        
        elif self.rad_ratio == 0.0:
            
            if field == 'toroidal' and l % 2 == 0:
                
                diag_mat[1:] += (1j * (-2*self.m) * self.eval_mat_odd + ek*lp2*self.orr_sqr_mat*self.eval_mat_odd)[1:]
                diag_mat[1:] += -lp * ek * self.df2_mat_odd[1:]
                
                diag_mat[0,:] = self.eval_mat_odd[0,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[1:]
                upper_mat[1:] += -upper_fac * self.df1_mat_even[1:]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:] += lower_fac * l*(self.orr_mat*self.eval_mat_even)[1:]
                lower_mat[1:] += -lower_fac * self.df1_mat_even[1:]
                
            elif field == 'toroidal':
                
                diag_mat[1:] += (1j * (-2*self.m) * self.eval_mat_even + ek*lp2*self.orr_sqr_mat*self.eval_mat_even)[1:]
                diag_mat[1:] += -lp * ek * self.df2_mat_even[1:]
                
                diag_mat[0,:] = self.eval_mat_even[0,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[1:]
                upper_mat[1:] += -upper_fac * self.df1_mat_odd[1:]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:] += lower_fac * l*(self.orr_mat*self.eval_mat_odd)[1:]
                lower_mat[1:] += -lower_fac * self.df1_mat_odd[1:]
                
                
            elif field == 'poloidal' and l % 2 == 0:
                
                diag_mat[2:] = 1j * (-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat_odd)[2:]
                diag_mat[2:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat_odd)[2:]
                diag_mat[2:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat_odd)[2:]
                diag_mat[2:] +=-1j * (-2*self.m)* self.df2_mat_odd[2:]
                diag_mat[2:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat_odd)[2:]
                diag_mat[2:] += lp * ek * self.df4_mat_odd[2:]
                
                diag_mat[0,:] = self.eval_mat_odd[0,:]
                diag_mat[1,:] = self.df1_mat_odd[0,:]


                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_even[2:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[2:] += lower_fac * l*(self.orr_mat*self.eval_mat_even)[2:]
                lower_mat[2:] += -lower_fac * self.df1_mat_even[2:]
                
            elif field == 'poloidal':
                
                diag_mat[2:] = 1j * (-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat_even)[2:]
                diag_mat[2:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat_even)[2:]
                diag_mat[2:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat_even)[2:]
                diag_mat[2:] +=-1j * (-2*self.m)* self.df2_mat_even[2:]
                diag_mat[2:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat_even)[2:]
                diag_mat[2:] += lp * ek * self.df4_mat_even[2:]
                
                diag_mat[0,:] = self.eval_mat_even[0,:]
                diag_mat[1,:] = self.df1_mat_even[0,:]


                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_odd[2:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[2:] += lower_fac * l*(self.orr_mat*self.eval_mat_odd)[2:]
                lower_mat[2:] += -lower_fac * self.df1_mat_odd[2:]
            
        return diag_mat,lower_mat,upper_mat

    def gen_block_row(self,field,l,for_freq,ek):
        
        lp = l*(l+1)
        lp2 = lp*lp
        
        lower_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        upper_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        diag_mat = sp.csr_matrix((self.N+1,self.N+1),dtype=complex)
        
        if self.rad_ratio > 0.0:
        
            if field == 'toroidal':
                
                diag_mat[1:-1,:] += (1j * (lp*for_freq-2*self.m) * self.eval_mat + ek*lp2*self.orr_sqr_mat*self.eval_mat)[1:-1,:]
                diag_mat[1:-1,:] += -lp * ek * self.df2_mat[1:-1,:]
                
                diag_mat[0,:] = self.eval_mat[0,:]
                diag_mat[-1,:] = self.eval_mat[-1,:]
                #diag_mat[-1,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[-1,:]
                
                upper_fac = 2*l*(l+2) * self.c_l[l-self.l_min+1]
                
                upper_mat[1:-1] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[1:-1]
                upper_mat[1:-1] += -upper_fac * self.df1_mat[1:-1]
    
    
                lower_fac = 2*(l-1)*(l+1) * self.c_l[l-self.l_min]
                
                lower_mat[1:-1] += lower_fac * l*(self.orr_mat*self.eval_mat)[1:-1]
                lower_mat[1:-1] += -lower_fac * self.df1_mat[1:-1]
                
                
            elif field == 'poloidal':
                
                diag_mat[2:-2,:] = 1j * (lp*for_freq-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat)[2:-2,:]
                diag_mat[2:-2,:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat)[2:-2,:]
                diag_mat[2:-2,:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat)[2:-2,:]
                diag_mat[2:-2,:] +=-1j * (lp*for_freq-2*self.m)* self.df2_mat[2:-2,:]
                diag_mat[2:-2,:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat)[2:-2,:]
                diag_mat[2:-2,:] += lp * ek * self.df4_mat[2:-2,:]
                
                diag_mat[0,:] = self.eval_mat[0,:]
                diag_mat[-1,:] = self.eval_mat[-1,:]
                diag_mat[1,:] = self.df1_mat[0,:]
                diag_mat[-2,:] = self.df1_mat[-1,:]
                #diag_mat[-2,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[-1,:]
                
                
                upper_fac = 2*l*(l+2) * self.c_l[l-self.l_min+1]
                
                upper_mat[2:-2,:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[2:-2,:]
                upper_mat[2:-2,:] += -upper_fac * self.df1_mat[2:-2,:]
                
            
                lower_fac = 2*(l-1)*(l+1) * self.c_l[l-self.l_min]
                
                lower_mat[2:-2,:] += lower_fac * l*(self.orr_mat*self.eval_mat)[2:-2,:]
                lower_mat[2:-2,:] += -lower_fac * self.df1_mat[2:-2,:]
                
            
        
        elif self.rad_ratio == 0.0:
            
            if field == 'toroidal' and l % 2 == 0:
                
                diag_mat[1:] += (1j * (lp*for_freq-2*self.m) * self.eval_mat_odd + ek*lp2*self.orr_sqr_mat*self.eval_mat_odd)[1:]
                diag_mat[1:] += -lp * ek * self.df2_mat_odd[1:]
                
                diag_mat[0,:] = self.eval_mat_odd[0,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[1:]
                upper_mat[1:] += -upper_fac * self.df1_mat_even[1:]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:] += lower_fac * l*(self.orr_mat*self.eval_mat_even)[1:]
                lower_mat[1:] += -lower_fac * self.df1_mat_even[1:]
                
            elif field == 'toroidal':
                
                diag_mat[1:] += (1j * (lp*for_freq-2*self.m) * self.eval_mat_even + ek*lp2*self.orr_sqr_mat*self.eval_mat_even)[1:]
                diag_mat[1:] += -lp * ek * self.df2_mat_even[1:]
                
                diag_mat[0,:] = self.eval_mat_even[0,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[1:]
                upper_mat[1:] += -upper_fac * self.df1_mat_odd[1:]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:] += lower_fac * l*(self.orr_mat*self.eval_mat_odd)[1:]
                lower_mat[1:] += -lower_fac * self.df1_mat_odd[1:]
                
                
            elif field == 'poloidal' and l % 2 == 0:
                
                diag_mat[2:] = 1j * (lp*for_freq-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat_odd)[2:]
                diag_mat[2:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat_odd)[2:]
                diag_mat[2:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat_odd)[2:]
                diag_mat[2:] +=-1j * (lp*for_freq-2*self.m)* self.df2_mat_odd[2:]
                diag_mat[2:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat_odd)[2:]
                diag_mat[2:] += lp * ek * self.df4_mat_odd[2:]
                
                diag_mat[0,:] = self.eval_mat_odd[0,:]
                diag_mat[1,:] = self.df1_mat_odd[0,:]


                upper_fac = 2*l*(l+2) * self.c_l[l+1-self.l_min]
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_even[2:]
                
            
                lower_fac = 2*(l-1)*(l+1) * self.c_l[l-self.l_min]
                
                lower_mat[2:] += lower_fac * l*(self.orr_mat*self.eval_mat_even)[2:]
                lower_mat[2:] += -lower_fac * self.df1_mat_even[2:]
                
            elif field == 'poloidal':
                
                diag_mat[2:] = 1j * (lp*for_freq-2*self.m) * lp * (self.orr_sqr_mat*self.eval_mat_even)[2:]
                diag_mat[2:] +=  lp2 * ek * (lp - 6) * (self.orr_sqr_mat * self.orr_sqr_mat*self.eval_mat_even)[2:]
                diag_mat[2:] += 4*lp2*ek * (self.orr_sqr_mat * self.orr_mat * self.df1_mat_even)[2:]
                diag_mat[2:] +=-1j * (lp*for_freq-2*self.m)* self.df2_mat_even[2:]
                diag_mat[2:] += -2*lp2*ek * (self.orr_sqr_mat * self.df2_mat_even)[2:]
                diag_mat[2:] += lp * ek * self.df4_mat_even[2:]
                
                diag_mat[0,:] = self.eval_mat_even[0,:]
                diag_mat[1,:] = self.df1_mat_even[0,:]


                upper_fac = 2*l*(l+2) * self.c_l[l+1-self.l_min]
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_odd[2:]
                
            
                lower_fac = 2*(l-1)*(l+1)* self.c_l[l-self.l_min]
                
                lower_mat[2:] += lower_fac * l*(self.orr_mat*self.eval_mat_odd)[2:]
                lower_mat[2:] += -lower_fac * self.df1_mat_odd[2:]
            
        return diag_mat,lower_mat,upper_mat


    
    
    def gen_freq_matrix(self,odd_flag,for_freq):
        
        if odd_flag == 'tor':
            
            odd_field = 'toroidal'
            even_field = 'poloidal'

        elif odd_flag == 'pol':
        
            even_field = 'toroidal'
            odd_field = 'poloidal'


        bmat_array = [[None for i in range(self.n_l)] for j in range(self.n_l)]
        
        
        l = self.l_min
        i = l-self.l_min
        
        if l % 2 == 1:
            
    
            diag_mat = self.gen_block_diag_freq(odd_field,l,for_freq)
            
        elif l % 2 == 0:
            
            diag_mat = self.gen_block_diag_freq(even_field,l,for_freq)
            
            
        bmat_array[i][i] = diag_mat

        
        
        for l in self.l_odd:
            
            i = l-self.l_min
            diag_mat= self.gen_block_diag_freq(odd_field,l,for_freq)
            
            bmat_array[i][i] = diag_mat

            
        for l in self.l_even:
            
            i = l-self.l_min
            diag_mat = self.gen_block_diag_freq(even_field,l,for_freq)
            
            bmat_array[i][i] = diag_mat

            
        l = self.l_max
        i = l-self.l_min
        if l % 2 == 1:
        
            diag_mat = self.gen_block_diag_freq(odd_field,l,for_freq)
            
            
        elif l % 2 == 0:
            
            diag_mat = self.gen_block_diag_freq(even_field,l,for_freq)
            
        
        bmat_array[i][i] = diag_mat

        
    
        return sp.bmat(bmat_array,format='csr')
        
    
    def gen_spatial_matrix(self,odd_flag,ek):
        
        if odd_flag == 'tor':
            
            odd_field = 'toroidal'
            even_field = 'poloidal'

        elif odd_flag == 'pol':
        
            even_field = 'toroidal'
            odd_field = 'poloidal'


        bmat_array = [[None for i in range(self.n_l)] for j in range(self.n_l)]
        
        
        l = self.l_min
        i = l-self.l_min
        
        if l % 2 == 1:
        
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(odd_field,l,ek)
            
        elif l % 2 == 0:
            
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(even_field,l,ek)
            
            
        bmat_array[i][i] = diag_mat
        bmat_array[i][i+1] = upper_mat
        
        
        for l in self.l_odd:
            
            i = l-self.l_min
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(odd_field,l,ek)
            
            bmat_array[i][i] = diag_mat
            bmat_array[i][i+1] = upper_mat
            bmat_array[i][i-1] = lower_mat
            
        for l in self.l_even:
            
            i = l-self.l_min
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(even_field,l,ek)
            
            bmat_array[i][i] = diag_mat
            bmat_array[i][i+1] = upper_mat
            bmat_array[i][i-1] = lower_mat
            
        l = self.l_max
        i = l-self.l_min
        if l % 2 == 1:
        
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(odd_field,l,ek)
            
            
        elif l % 2 == 0:
            
            diag_mat,lower_mat,upper_mat = self.gen_block_row_spatial(even_field,l,ek)
            
        
        bmat_array[i][i] = diag_mat
        bmat_array[i][i-1] = lower_mat
        
    
        return sp.bmat(bmat_array,format='csr')
    
    
    def gen_PDE_matrix(self,odd_flag,for_freq,ek):
        
        if odd_flag == 'tor':
            
            odd_field = 'toroidal'
            even_field = 'poloidal'

        elif odd_flag == 'pol':
        
            even_field = 'toroidal'
            odd_field = 'poloidal'


        bmat_array = [[None for i in range(self.n_l)] for j in range(self.n_l)]
        
        
        l = self.l_min
        i = l-self.l_min
        
        if l % 2 == 1:
        
            diag_mat,lower_mat,upper_mat = self.gen_block_row(odd_field,l,for_freq,ek)
            
        elif l % 2 == 0:
            
            diag_mat,lower_mat,upper_mat = self.gen_block_row(even_field,l,for_freq,ek)
            
            
        bmat_array[i][i] = diag_mat
        bmat_array[i][i+1] = upper_mat
        
        
        for l in self.l_odd:
            
            i = l-self.l_min
            diag_mat,lower_mat,upper_mat = self.gen_block_row(odd_field,l,for_freq,ek)
            
            bmat_array[i][i] = diag_mat
            bmat_array[i][i+1] = upper_mat
            bmat_array[i][i-1] = lower_mat
            
        for l in self.l_even:
            
            i = l-self.l_min
            diag_mat,lower_mat,upper_mat = self.gen_block_row(even_field,l,for_freq,ek)
            
            bmat_array[i][i] = diag_mat
            bmat_array[i][i+1] = upper_mat
            bmat_array[i][i-1] = lower_mat
            
        l = self.l_max
        i = l-self.l_min
        if l % 2 == 1:
        
            diag_mat,lower_mat,upper_mat = self.gen_block_row(odd_field,l,for_freq,ek)
            
            
        elif l % 2 == 0:
            
            diag_mat,lower_mat,upper_mat = self.gen_block_row(even_field,l,for_freq,ek)
            
        
        bmat_array[i][i] = diag_mat
        bmat_array[i][i-1] = lower_mat
        
    
        return sp.bmat(bmat_array,format='csr')
        
class Soln_forced:
    
    def __init__(self,soln,mat_builder):
        
        self.mb = mat_builder
        self.soln = soln
    

    def process_soln(self,odd_flag):
        

        tor_arr = np.zeros((self.mb.n_l,self.mb.N+1),dtype=complex)
        dr_tor_arr = np.zeros_like(tor_arr)
        pol_arr = np.zeros((self.mb.n_l,self.mb.N+1),dtype=complex)
        dr_pol_arr = np.zeros((self.mb.n_l,self.mb.N+1),dtype=complex)
        dr2_pol_arr = np.zeros_like(tor_arr)
        
        if self.mb.rad_ratio > 0.0:
            if odd_flag == 'tor':
                
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
        elif self.mb.rad_ratio == 0.0:
            
            if odd_flag == 'tor':
                
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat_even @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat_even @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat_odd @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat_odd @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat_odd @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat_odd @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat_odd @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat_even @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat_even @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat_even @ self.soln[i*(self.mb.N+1):(i+1)*(self.mb.N+1)]
            
        return tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr
        


    def calc_vel_field(self,field,tor_arr,pol_arr,dr_pol_arr,theta_grid,l_max):
        
        n_theta = len(theta_grid)
        vel_arr = np.zeros((self.N+1,n_theta),dtype=complex)
        orr = 1/self.r_grid
        orr_sqr = orr * orr
        
        s_theta = np.sin(theta_grid)
        oos_theta = 1/s_theta
        
        if field == 'radial':
            
            for l in range(self.l_min,l_max+1):
                
                i = l - self.l_min
                
                vel_arr += l*(l+1) * orr_sqr[:,np.newaxis] * pol_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
            
            
        if field == 'colatitudinal':
            
            
            for l in range(self.l_min,l_max+1):
                        
                i = l - self.l_min
                vel_arr +=oos_theta[np.newaxis,:] * orr[:,np.newaxis] * dr_pol_arr[i,:,np.newaxis] * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                vel_arr += oos_theta[np.newaxis,:] * orr[:,np.newaxis] * tor_arr[i,:,np.newaxis] * 1j * self.m *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
        
        
        if field == 'longitudinal':
            
            for l in range(self.l_min,l_max+1):
                        
                i = l - self.l_min
                    
                vel_arr +=-oos_theta[np.newaxis,:] * orr[:,np.newaxis] * tor_arr[i,:,np.newaxis] * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                vel_arr += oos_theta[np.newaxis,:] * orr[:,np.newaxis] * dr_pol_arr[i,:,np.newaxis] * 1j * self.m *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
        
        
        return vel_arr
    
    

    
    def calc_vel_grad(self,field,grad,tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,l_max):
        
        n_theta = len(theta_grid)
        arr = np.zeros((self.N+1,n_theta),dtype=complex)
        orr = 1/self.r_grid
        orr_sqr = orr * orr
        
        s_theta = np.sin(theta_grid)
        oos_theta = 1/s_theta
        
        if field == 'radial':
            
            if grad == 'radial':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += l *(l+1)  * (dr_pol_arr[i,:,np.newaxis] - 2*orr[:,np.newaxis]*pol_arr[i,:,np.newaxis]) * sphrharm(l,self.m,theta_grid,0)
                
                arr *= orr_sqr[:,np.newaxis]
                
            if grad == 'colatitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += (l*(l+1)*orr[:,np.newaxis]*pol_arr[i,:,np.newaxis] - dr_pol_arr[i,:,np.newaxis]) * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                    arr += -1j*self.m * tor_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                
                arr *= orr_sqr[:,np.newaxis] * oos_theta[np.newaxis,:]
            
            if grad == 'longitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += 1j*self.m * (l*(l+1) * orr[:,np.newaxis]*pol_arr[i,:,np.newaxis] - dr_pol_arr[i,:,np.newaxis]) * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                    arr += tor_arr[i,:,np.newaxis] * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                
                arr *= orr_sqr[:,np.newaxis] * oos_theta[np.newaxis,:]
            
        if field == 'colatitudinal':
            
            if grad == 'radial':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += (dr2_pol_arr[i,:,np.newaxis] - orr[:,np.newaxis]*dr_pol_arr[i,:,np.newaxis]) * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                    arr += 1j*self.m * (dr_tor_arr[i,:,np.newaxis]-orr[:,np.newaxis]*tor_arr[i,:,np.newaxis]) * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                    
                arr *= orr[:,np.newaxis] * oos_theta[np.newaxis,:]
            
            if grad == 'colatitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += dr_pol_arr[i,:,np.newaxis] * -np.cos(theta_grid[np.newaxis,:])*(l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                    arr += dr_pol_arr[i,:,np.newaxis] * l * np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3)) * ((l+1) * np.sqrt((l+2-self.m)*(l+2+self.m)/(2*l+3)/(2*l+5))*sphrharm(l+2,self.m,theta_grid,0)[np.newaxis,:]-(l+2)*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+3)/(2*l+1)) *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:])
                    arr += -dr_pol_arr[i,:,np.newaxis] * np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1)) * (l+1) * ((l-1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]-l*np.sqrt((l-1-self.m)*(l-1+self.m)/(2*l-1)/(2*l-3))*sphrharm(l-2,self.m,theta_grid,0)[np.newaxis,:])
                    
                    arr += 1j*self.m * tor_arr[i,:,np.newaxis] * ((l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])-np.cos(theta_grid[np.newaxis,:])*sphrharm(l,self.m,theta_grid,0)[np.newaxis,:])
                    
                    arr += l*(l+1) * np.sin(theta_grid[np.newaxis,:])**2 *orr[:,np.newaxis] * pol_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                    
                arr *= orr[:,np.newaxis]**2 * oos_theta[np.newaxis,:]**2
            
            if grad == 'longitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += self.m * (dr_pol_arr[i,:,np.newaxis]*1j * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])-self.m*tor_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:])
                    arr += -np.cos(theta_grid[np.newaxis,:])*(dr_pol_arr[i,:,np.newaxis]*1j*self.m *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]-tor_arr[i,:,np.newaxis] *(l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:]))
                
                arr *= orr[:,np.newaxis]**2 * oos_theta[np.newaxis,:]**2
            
        if field == 'longitudinal':
            
            if grad == 'radial':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += 1j*self.m * (dr2_pol_arr[i,:,np.newaxis] - orr[:,np.newaxis] * dr_pol_arr[i,:,np.newaxis]) * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                    arr += -(dr_tor_arr[i,:,np.newaxis] - orr[:,np.newaxis] * tor_arr[i,:,np.newaxis] ) * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                
                arr *= orr[:,np.newaxis] * oos_theta[np.newaxis,:]
            
            if grad == 'colatitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += -tor_arr[i,:,np.newaxis] * -np.cos(theta_grid[np.newaxis,:])*(l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])
                    arr += -tor_arr[i,:,np.newaxis] * l * np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3)) * ((l+1) * np.sqrt((l+2-self.m)*(l+2+self.m)/(2*l+3)/(2*l+5))*sphrharm(l+2,self.m,theta_grid,0)[np.newaxis,:]-(l+2)*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+3)/(2*l+1)) *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:])
                    arr += tor_arr[i,:,np.newaxis] * np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1)) * (l+1) * ((l-1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]-l*np.sqrt((l-1-self.m)*(l-1+self.m)/(2*l-1)/(2*l-3))*sphrharm(l-2,self.m,theta_grid,0)[np.newaxis,:])
                
                    arr += 1j*self.m * ((l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])-np.cos(theta_grid[np.newaxis,:])*sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]) * dr_pol_arr[i,:,np.newaxis]
                
                arr *= orr[:,np.newaxis]**2 * oos_theta[np.newaxis,:]**2
                
            if grad == 'longitudinal':
                
                for l in range(self.l_min,l_max+1):
                    
                    i = l - self.l_min
                    
                    arr += self.m * (-self.m * dr_pol_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]-1j* (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:])*tor_arr[i,:,np.newaxis])
                    arr += np.cos(theta_grid[np.newaxis,:]) * ( dr_pol_arr[i,:,np.newaxis] * (l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:]) + 1j*self.m *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]*tor_arr[i,:,np.newaxis])
                    
                    arr += l*(l+1) * np.sin(theta_grid[np.newaxis,:])**2 *orr[:,np.newaxis] * pol_arr[i,:,np.newaxis] * sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]
                    
                arr *= orr[:,np.newaxis]**2 * oos_theta[np.newaxis,:]**2
                
                
            
        return arr
    


class Rhs_builder:
    
    def __init__(self,N,rad_ratio,m,l_max):
        
        self.N = N
        self.rad_ratio = rad_ratio
        self.m = m
        self.l_max = l_max
        
    
        self.l_min,self.n_l,self.l_odd,self.l_even,self.l_even_full,self.l_odd_full = l_setup(self.m,self.l_max)
        
        
    def gen_rhs(self,bc_list,odd_flag):
        
        self.gen_bc_arrs(bc_list)
        
        self.rhs_tor_odd = np.zeros(self.n_l*(self.N+1),dtype=complex)
        self.rhs_tor_even = np.zeros_like(self.rhs_tor_odd)
        
    
        for l in self.l_odd_full:
            
            i = l - self.l_min
            self.rhs_tor_odd[i*(self.N+1)] = self.tor_t[i]
            self.rhs_tor_odd[(i+1)*(self.N+1)-1] = self.tor_b[i]
            
            self.rhs_tor_even[i*(self.N+1)] = self.pol_t[i]
            self.rhs_tor_even[i*(self.N+1)+1] = self.dr_pol_t[i]
            self.rhs_tor_even[(i+1)*(self.N+1)-1] = self.pol_b[i]
            self.rhs_tor_even[(i+1)*(self.N+1)-2] = self.dr_pol_b[i]
            
        for l in self.l_even_full:
            
            i = l - self.l_min
            self.rhs_tor_even[i*(self.N+1)] = self.tor_t[i]
            self.rhs_tor_even[(i+1)*(self.N+1)-1] = self.tor_b[i]
            
            self.rhs_tor_odd[i*(self.N+1)] = self.pol_t[i]
            self.rhs_tor_odd[i*(self.N+1)+1] = self.dr_pol_t[i]
            self.rhs_tor_odd[(i+1)*(self.N+1)-1] = self.pol_b[i]
            self.rhs_tor_odd[(i+1)*(self.N+1)-2] = self.dr_pol_b[i]
    
        if odd_flag == 'tor':
            
            self.rhs = self.rhs_tor_odd
        
        elif odd_flag == 'pol':
        
            self.rhs = self.rhs_tor_even
        

    def gen_bc_arrs(self,bc_list):
        
        # bc_list = list of lists with each sublist formatted as: ['tor/pol/dr_pol','t/b','l',value]
        
        self.pol_t = np.zeros(self.n_l*(self.N+1),dtype=complex)
        self.pol_b = np.zeros_like(self.pol_t)
        self.tor_t = np.zeros_like(self.pol_t)
        self.tor_b = np.zeros_like(self.pol_t)
        self.dr_pol_t = np.zeros_like(self.pol_t)
        self.dr_pol_b = np.zeros_like(self.pol_t)
        
        for bc in bc_list:
            
            field,tb,l,val = bc
            
            i = l - self.l_min
            
            if i < 0 or i >= self.n_l: print('warning: invalid l value to gen_bc_arrs...')
            
            if field == 'pol':
                
                if tb == 't':
                    
                    self.pol_t[i] = val
                    
                elif tb == 'b':
                    
                    self.pol_b[i] = val
                
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                
            elif field == 'tor':
                
                if tb == 't':
                    
                    self.tor_t[i] = val
                    
                elif tb == 'b':
                    
                    self.tor_b[i] = val
                    
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                    
            elif field == 'dr_pol':
                
                if tb == 't':
                    
                    self.dr_pol_t[i] = val
                    
                elif tb == 'b':
                    
                    self.dr_pol_b[i] = val
                    
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                    
            else:
                
                print('warning: invalid field value to gen_bc_arrs...')
                

   





