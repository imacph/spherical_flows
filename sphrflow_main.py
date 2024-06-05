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


        
class LN:
    
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
            self.r_grid = self.x_grid * (self.r_end - self.r_start)/2 + (self.r_end + self.r_start)/2
            self.eval_mat,self.df1_mat,self.df2_mat,self.df4_mat = self.gen_deriv_mats()
            
        elif self.rad_ratio == 0.0:
            self.x_grid = np.cos(np.linspace(0,N,N+1)*np.pi/(2*N+1))
            self.r_grid = self.x_grid
            self.eval_mat_even,self.eval_mat_odd,self.df1_mat_even,self.df1_mat_odd,self.df2_mat_even,self.df2_mat_odd,self.df4_mat_even,self.df4_mat_odd = self.gen_deriv_mats()
        
        if self.m == 0:
            
            self.l_min = 1
        
        else:
            
            self.l_min = self.m
        
        if self.l_min % 2 == 0:
            
            self.l_odd = [i for i in range(self.l_min+1,self.l_max,2)]
            self.l_even = [i for i in range(self.l_min+2,self.l_max,2)]
        
        else:
            
            self.l_odd = [i for i in range(self.l_min+2,self.l_max,2)]
            self.l_even = [i for i in range(self.l_min+1,self.l_max,2)]
            
        if self.l_min % 2 == 0:
            
            self.l_even_full = [self.l_min] + self.l_even
            self.l_odd_full = self.l_odd
        else:
            
            self.l_odd_full = [self.l_min] + self.l_odd
            self.l_even_full = self.l_even
            
        if self.l_max % 2 == 0:
            
            self.l_even_full = self.l_even_full + [self.l_max]
        
        else:
            
            self.l_odd_full = self.l_odd_full + [self.l_max]
            
            
        self.n_l = self.l_max - self.l_min + 1
        
        self.orr_mat = sp.csr_matrix(np.diag(1/self.r_grid),dtype=complex)
        self.orr_sqr_mat = self.orr_mat * self.orr_mat
            

            
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
            
            for i in range(self.N+1):

                
                grid_diag = sp.csr_matrix(np.diag(self.x_grid),dtype=complex)
                grid_sqr_diag = grid_diag * grid_diag
                n = 2*i+1
                
                eval_mat[:,i] = np.cos(n*np.arccos(self.x_grid))
                
                df1_mat[0,i] = n*n
                df2_mat[0,i] = n*n * (n*n-1)/3
                df3_mat[0,i] = n*n * (n*n-1)/3 * (n*n-4)/5
                df4_mat[0,i] = n*n * (n*n-1)/3 * (n*n-4)/5 * (n*n-9)/7

                df1_mat[1:,i] = n * np.sin(n*np.arccos(self.x_grid[1:]))/np.sin(np.arccos(self.x_grid[1:]))
                df2_mat[1:,i] = n*self.x_grid[1:]*np.sin(n*np.arccos(self.x_grid[1:]))/(1-self.x_grid[1:]**2)**(3/2) - n*n * np.cos(n*np.arccos(self.x_grid[1:]))/(1-self.x_grid[1:]**2)
                df3_mat[1:,i] = n/(1-self.x_grid[1:])**(5/2) * ((n**2*(self.x_grid[1:]**2-1)+2*self.x_grid[1:]**2+1)*np.sin(n*np.arccos(self.x_grid[1:]))-3*n*self.x_grid[1:]*np.sqrt(1-self.x_grid[1:]**2)*np.cos(n*np.arccos(self.x_grid[1:])))
                
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
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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
                #diag_mat[-1,:] = self.eval_mat[-1,:]
                diag_mat[-1,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[-1,:]
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:-1] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[1:-1]
                upper_mat[1:-1] += -upper_fac * self.df1_mat[1:-1]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
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
                #diag_mat[-2,:] = self.df1_mat[-1,:]
                diag_mat[-2,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[-1,:]
                
                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:-2,:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat)[2:-2,:]
                upper_mat[2:-2,:] += -upper_fac * self.df1_mat[2:-2,:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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


                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_even[2:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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


                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_odd[2:]
                
            
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+m)*(l-m)/(2*l-1)/(2*l+1))
                
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
        

    def process_soln(self,odd_flag,soln):
        

        tor_arr = np.zeros((self.n_l,self.N+1),dtype=complex)
        dr_tor_arr = np.zeros_like(tor_arr)
        pol_arr = np.zeros((self.n_l,self.N+1),dtype=complex)
        dr_pol_arr = np.zeros((self.n_l,self.N+1),dtype=complex)
        dr2_pol_arr = np.zeros_like(tor_arr)
        
        if self.rad_ratio > 0.0:
            if odd_flag == 'tor':
                
                for l in self.l_odd_full:
                    
                    i = l-self.l_min
                    tor_arr[i,:] = self.eval_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_tor_arr[i,:] = self.df1_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
                for l in self.l_even_full:
                    
                    i = l-self.l_min
                    pol_arr[i,:] = self.eval_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_pol_arr[i,:] = self.df1_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr2_pol_arr[i,:] = self.df2_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.l_even_full:
                    
                    i = l-self.l_min
                    tor_arr[i,:] = self.eval_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_tor_arr[i,:] = self.df1_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
                for l in self.l_odd_full:
                    
                    i = l-self.l_min
                    pol_arr[i,:] = self.eval_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_pol_arr[i,:] = self.df1_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr2_pol_arr[i,:] = self.df2_mat @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
        elif self.rad_ratio == 0.0:
            
            if odd_flag == 'tor':
                
                for l in self.l_odd_full:
                    
                    i = l-self.l_min
                    tor_arr[i,:] = self.eval_mat_even @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_tor_arr[i,:] = self.df1_mat_even @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
                for l in self.l_even_full:
                    
                    i = l-self.l_min
                    pol_arr[i,:] = self.eval_mat_odd @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_pol_arr[i,:] = self.df1_mat_odd @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr2_pol_arr[i,:] = self.df2_mat_odd @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.l_even_full:
                    
                    i = l-self.l_min
                    tor_arr[i,:] = self.eval_mat_odd @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_tor_arr[i,:] = self.df1_mat_odd @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    
                for l in self.l_odd_full:
                    
                    i = l-self.l_min
                    pol_arr[i,:] = self.eval_mat_even @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr_pol_arr[i,:] = self.df1_mat_even @ soln[i*(self.N+1):(i+1)*(self.N+1)]
                    dr2_pol_arr[i,:] = self.df2_mat_even @ soln[i*(self.N+1):(i+1)*(self.N+1)]
            
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
                    arr += -(np.cos(theta_grid[np.newaxis,:])*dr_pol_arr[i,:,np.newaxis]*1j*m *sphrharm(l,self.m,theta_grid,0)[np.newaxis,:]-tor_arr[i,:,np.newaxis] *(l*np.sqrt((l+1-self.m)*(l+1+self.m)/(2*l+1)/(2*l+3))* sphrharm(l+1,self.m,theta_grid,0)[np.newaxis,:]-(l+1)*np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))*sphrharm(l-1,self.m,theta_grid,0)[np.newaxis,:]))
                
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
    def gen_rhs(self,tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b):
        
        rhs_tor_odd = np.zeros(self.n_l*(self.N+1),dtype=complex)
        rhs_tor_even = np.zeros_like(rhs_tor_odd)
        
    
        for l in self.l_odd_full:
            
            i = l - self.l_min
            rhs_tor_odd[i*(self.N+1)] = tor_t[i]
            rhs_tor_odd[(i+1)*(self.N+1)-1] = tor_b[i]
            
            rhs_tor_even[i*(self.N+1)] = pol_t[i]
            rhs_tor_even[i*(self.N+1)+1] = dr_pol_t[i]
            rhs_tor_even[(i+1)*(self.N+1)-1] = pol_b[i]
            rhs_tor_even[(i+1)*(self.N+1)-2] = dr_pol_b[i]
            
        for l in self.l_even_full:
            
            i = l - self.l_min
            rhs_tor_even[i*(self.N+1)] = tor_t[i]
            rhs_tor_even[(i+1)*(self.N+1)-1] = tor_b[i]
            
            rhs_tor_odd[i*(self.N+1)] = pol_t[i]
            rhs_tor_odd[i*(self.N+1)+1] = dr_pol_t[i]
            rhs_tor_odd[(i+1)*(self.N+1)-1] = pol_b[i]
            rhs_tor_odd[(i+1)*(self.N+1)-2] = dr_pol_b[i]
            
            
        return rhs_tor_odd,rhs_tor_even

    def gen_bc_arrs(self,bc_list):
        
        # bc_list = list of lists with each sublist formatted as: ['tor/pol/dr_pol','t/b','l',value]
        
        pol_t = np.zeros(self.n_l*(self.N+1),dtype=complex)
        pol_b = np.zeros_like(pol_t)
        tor_t = np.zeros_like(pol_t)
        tor_b = np.zeros_like(pol_t)
        dr_pol_t = np.zeros_like(pol_t)
        dr_pol_b = np.zeros_like(pol_t)
        
        for bc in bc_list:
            
            field,tb,l,val = bc
            
            i = l - self.l_min
            
            if i < 0 or i >= self.n_l: print('warning: invalid l value to gen_bc_arrs...')
            
            if field == 'pol':
                
                if tb == 't':
                    
                    pol_t[i] = val
                    
                elif tb == 'b':
                    
                    pol_b[i] = val
                
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                
            elif field == 'tor':
                
                if tb == 't':
                    
                    tor_t[i] = val
                    
                elif tb == 'b':
                    
                    tor_b[i] = val
                    
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                    
            elif field == 'dr_pol':
                
                if tb == 't':
                    
                    dr_pol_t[i] = val
                    
                elif tb == 'b':
                    
                    dr_pol_b[i] = val
                    
                else:
                    
                    print('warning: invalid t/b value to gen_bc_arrs...')
                    
            else:
                
                print('warning: invalid field value to gen_bc_arrs...')
                
        return tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b
   

def Ekman_disp(freq,eps,E,eta):
    
    return 2*np.pi/15/np.sqrt(2) * np.abs(eps)**2 * E**(1/2)/(1-eta)**4 * ((2-freq)**(5/2)+(2+freq)**(5/2)-1/7*((2-freq)**(7/2)+(2+freq)**(7/2)))


def Ekman_kin(freq,eps,E,eta):
    
    return np.pi/3/np.sqrt(2) * eps**2*E**(1/2)/(1-eta)**4 * ((2-freq)**(3/2)+(2+freq)**(3/2)-1/5*((2-freq)**(5/2)+(2+freq)**(5/2)))


    
N = 113
l_max = 70
rad_ratio = 0.
m = 2

for_freq = 1.232
ek = 1e-7
bc_list = [['pol','t',2,2/3*np.sqrt(2*np.pi/15)/(1-rad_ratio)**2]]


t0 = time()
LN_case = LN(N,rad_ratio,m,l_max)

PDE_mat = LN_case.gen_PDE_matrix('tor',for_freq,ek)

LU = spla.splu(PDE_mat)

tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b = LN_case.gen_bc_arrs(bc_list)

rhs_tor_odd,rhs_tor_even = LN_case.gen_rhs(tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b)
soln0 = LU.solve(rhs_tor_odd)


freq_mat_1 = LN_case.gen_freq_matrix('tor',1.)
print(time()-t0)

n_freq = 100
marg = 3*np.sqrt(ek)
freq_arr = np.linspace(for_freq-marg,for_freq+marg,n_freq)
en_arr = np.zeros_like(freq_arr)

from tqdm import tqdm

for k in tqdm(range(n_freq)):
    freq_mat = (freq_arr[k]-for_freq)*freq_mat_1


    soln = soln0
    n_it = 100
    res = np.inf
    tol = 1e-4
    i = 0
    
    while i < n_it and res >= tol:
        
        
        soln = LU.solve(rhs_tor_odd-freq_mat @ soln)
        res = np.linalg.norm((PDE_mat+freq_mat) @ soln-rhs_tor_odd)
        
        i += 1
    
        
    
    tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr = LN_case.process_soln('tor',soln)
    

    n_theta = N
    theta_min = 0
    theta_max = np.pi/2
    theta_grid = 0.5 * (np.cos(np.linspace(1,n_theta,n_theta)*np.pi/(n_theta+1))[::-1] * (theta_max-theta_min) + (theta_max+theta_min))
    
    q_phi = LN_case.calc_vel_field('longitudinal',tor_arr,pol_arr,dr_pol_arr,theta_grid,LN_case.l_max)
    q_r = LN_case.calc_vel_field('radial',tor_arr,pol_arr,dr_pol_arr,theta_grid,LN_case.l_max)
    q_theta = LN_case.calc_vel_field('colatitudinal',tor_arr,pol_arr,dr_pol_arr,theta_grid,LN_case.l_max)
    
    en = 1/4 * (np.abs(q_phi)**2+np.abs(q_r)**2+np.abs(q_theta)**2)
    
    tot_en = 4*np.pi * np.trapz(np.sin(theta_grid)*np.trapz(LN_case.r_grid[::-1,np.newaxis]**2*en,x=LN_case.r_grid[::-1],axis=0),x=theta_grid,axis=0)
    
    en_arr[k] = tot_en

np.savetxt('freqs_it_7.txt',freq_arr)
np.savetxt('ens_it_7.txt',en_arr)


'''
dr_q_r = LN_case.calc_vel_grad('radial','radial',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dtheta_q_r = LN_case.calc_vel_grad('radial','colatitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dphi_q_r = LN_case.calc_vel_grad('radial','longitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)

dr_q_theta = LN_case.calc_vel_grad('colatitudinal','radial',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dtheta_q_theta = LN_case.calc_vel_grad('colatitudinal','colatitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dphi_q_theta = LN_case.calc_vel_grad('colatitudinal','longitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)

dr_q_phi = LN_case.calc_vel_grad('longitudinal','radial',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dtheta_q_phi = LN_case.calc_vel_grad('longitudinal','colatitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)
dphi_q_phi = LN_case.calc_vel_grad('longitudinal','longitudinal',tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr,theta_grid,LN_case.l_max)


disp = 2*np.abs(dr_q_r)**2 + np.abs(dr_q_theta)**2 + np.abs(dr_q_phi)**2
disp +=  np.abs(dtheta_q_r)**2 + 2*np.abs(dtheta_q_theta)**2 + np.abs(dtheta_q_phi)**2
disp +=  np.abs(dphi_q_r)**2 + np.abs(dphi_q_theta)**2 + 2*np.abs(dphi_q_phi)**2

disp += np.real(dr_q_theta * np.conjugate(dtheta_q_r)) + np.real(dr_q_phi * np.conjugate(dphi_q_r)) 
disp += np.real(dtheta_q_r * np.conjugate(dr_q_theta)) + np.real(dtheta_q_phi * np.conjugate(dphi_q_theta)) 
disp += np.real(dphi_q_r * np.conjugate(dr_q_phi)) + np.real(dphi_q_theta * np.conjugate(dtheta_q_phi)) 

disp *= 0.5


tot_disp = ek*2*2*np.pi * np.trapz(np.sin(theta_grid) * np.trapz(LN_case.r_grid[::-1,np.newaxis]**2*disp[::-1,:],x=LN_case.r_grid[::-1],axis=0),x=theta_grid,axis=0)

print(tot_disp)
print(tot_disp/Ekman_disp(for_freq,1,ek,rad_ratio))
print(Ekman_disp(for_freq,1,ek,rad_ratio))



tau_phi_r =  dphi_q_r + dr_q_phi 
tau_theta_r = dtheta_q_r + dr_q_theta


surf_power = 4*np.pi * ek* LN_case.r_end**2* np.trapz(np.sin(theta_grid) * 0.5*np.real(q_phi[0,:]*np.conjugate(tau_phi_r[0,:])+q_theta[0,:]*np.conjugate(tau_theta_r[0,:])),x=theta_grid,axis=0)

print(surf_power)
print(surf_power/Ekman_disp(for_freq,1,ek,rad_ratio))

fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=200)

if rad_ratio > 0.0:
    ss = np.array([[LN_case.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[LN_case.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])

if rad_ratio == 0.0:
    ss = np.array([[LN_case.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[LN_case.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])

field = np.real(q_phi)


vmin = np.min(field)
vmax = np.max(field)



if -vmin > vmax:
    
    vmax1 = -vmin
    vmin1 = vmin
    
elif -vmin <= vmax:
    
    vmin1 = -vmax
    vmax1 = vmax


min_dec = int(np.ceil(np.log10(-vmin1)))
max_dec = int(np.ceil(np.log10(vmax1)))

thres = max(abs(vmin1),abs(vmax1))/15


print(vmin,vmax)


if vmin < 0:
    norm = colors.SymLogNorm(vmin = vmin1, vmax = vmax1, linthresh=thres)
    levels = -np.logspace(np.log10(thres),min_dec,100)[::-1]
    levels = np.append(levels,np.linspace(-thres,thres,100)[1:])
    levels = np.append(levels,np.logspace(np.log10(thres),max_dec,100)[1:])
    cmap = 'seismic'
elif vmin >= 0:
    
    
    off=-13
    top = -2
    norm = colors.LogNorm(vmin = 10**(off)*vmax,vmax=10**(top)*vmax)
    levels = np.logspace(max_dec+off,max_dec+top,100)


    cmap = 'inferno'
step = 1
#norm = colors.TwoSlopeNorm(vcenter=0,vmin=vmin,vmax=vmax)
p=ax.contourf(ss[::step,::step],zz[::step,::step],field[::step,::step],levels=levels,cmap=cmap,norm=norm)
cbar = fig.colorbar(p)
#cbar.set_ticks([vmin,0,vmax])
#cbar.ax.set_yticklabels([round(vmin,5),0,round(vmax,5)])

ax.set_aspect('equal')
ax.axis('off')

'''



