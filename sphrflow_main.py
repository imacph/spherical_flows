import numpy as np
from scipy import sparse as sp
from scipy.special import lpmv
import scipy.sparse.linalg as spla

def gen_theta_grid_cheb(n_theta,theta_min,theta_max):
    
    return 0.5 * (np.cos(np.linspace(1,n_theta,n_theta)*np.pi/(n_theta+1))[::-1] * (theta_max-theta_min) + (theta_max+theta_min))

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
        
        self.mb = Matrix_builder_forced(N,rad_ratio,m,l_max)
        self.rhs_builder = Boundary_rhs_builder(N,rad_ratio,m,l_max)

    def iterate(self,freq_array,ek,bc_list,odd_flag,tol=1e-4,max_inner_it=30,savepath = None):
        

        base_freq_mat = self.mb.gen_freq_matrix(odd_flag,1.)
        self.rhs_builder.gen_rhs(bc_list,odd_flag)
        
        
        n_freq = len(freq_array)
        powers = np.zeros((n_freq,2))
        k = 0 
        while k <= n_freq-1:
            print(k,k/len(freq_array))
            freq0 = freq_array[k]
            
            PDE_mat = PDE_matrix_frame(self.mb.gen_PDE_matrix(odd_flag,freq_array[k],ek),self.mb,ek,freq0)
            PDE_soln0 = PDE_mat.solve_sys(self.rhs_builder.rhs)
            PDE_soln0.process_soln('tor')
            
            spat_rep = Spatial_representation(gen_theta_grid_cheb(2*self.mb.n_l,0,np.pi), PDE_soln0)
            
            spat_rep.calc_surface_power()

            powers[k,1] = spat_rep.power_cmb
            powers[k,0] = freq_array[k]
            inner_iterating = True
            n_suc_it = 0
            while inner_iterating:
                
                inner_k = 0
                res = np.inf
                PDE_soln = PDE_soln0
                while inner_k < max_inner_it-1 and res > tol and k+n_suc_it+1 <= n_freq-1:
                    
                    freq = freq_array[k+1+n_suc_it]
                    
                    freq_mat = (freq-freq_array[k])*base_freq_mat
                    
                    PDE_soln = PDE_mat.solve_sys(self.rhs_builder.rhs-freq_mat @ PDE_soln.soln)
                    res = np.linalg.norm((PDE_mat.matrix+freq_mat) @ PDE_soln.soln-self.rhs_builder.rhs)
                    
                    inner_k += 1
            
                if inner_k == max_inner_it-1 and res > tol:
                    
                    inner_iterating = False
                    
                else:
                    #print('iter',n_suc_it,inner_k,res,freq)
                    
                    PDE_soln.process_soln('tor')
                    spat_rep.s=PDE_soln
                    
                    spat_rep.calc_surface_power()
                    powers[k+1+n_suc_it,1] = spat_rep.power_cmb
                    powers[k+1+n_suc_it,0] = freq
                    n_suc_it += 1

            k += n_suc_it +1
            
            if savepath != None:
                np.savetxt(savepath,powers)
                    
            

        
        
class Matrix_builder_forced:
    
    def __init__(self,N,rad_ratio,m,l_max,stress_free_icb = False,stress_free_cmb=False):
        
        self.N = N
        self.rad_ratio = rad_ratio
        self.m = m
        self.l_max = l_max
        
        self.r_end = 1/(1-rad_ratio)
        self.r_start = self.r_end * rad_ratio
        
        self.r_fac = 2/(self.r_end-self.r_start)
        
        self.stress_free_icb = stress_free_icb
        self.stress_free_cmb = stress_free_cmb
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
                
                
                if self.stress_free_icb: 
                    diag_mat[-1,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[-1,:]
                else:
                    diag_mat[-1,:] = self.eval_mat[-1,:]
                    
                if self.stress_free_cmb:
                    diag_mat[0,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[0,:]
                else:
                    diag_mat[0,:] = self.eval_mat[0,:]
                    
                    
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
                
                
                if self.stress_free_icb:
                    diag_mat[-2,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[-1,:]
                else:
                    diag_mat[-2,:] = self.df1_mat[-1,:]
                    
                if self.stress_free_cmb:
                    diag_mat[1,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[0,:]
                
                else:
                    diag_mat[1,:] = self.df1_mat[0,:]
                    
                    
                diag_mat[0,:] = self.eval_mat[0,:]
                diag_mat[-1,:] = self.eval_mat[-1,:]
                
                
                
                
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
    
    def __init__(self,soln,mat_builder,ek,for_freq):
        
        self.mb = mat_builder
        self.soln = soln
        self.ek = ek
        self.for_freq = for_freq
        

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
            
        self.tor_arr,self.dr_tor_arr,self.pol_arr,self.dr_pol_arr,self.dr2_pol_arr= tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr
        




    def calc_vel_field(self,spatial_rep):
        

        spatial_rep.q_r = np.tensordot(spatial_rep.sphrharm_eval_mat,self.pol_arr*(spatial_rep.l_arr*(spatial_rep.l_arr+1))[:,np.newaxis],axes=1).T/self.mb.r_grid[:,np.newaxis]**2
        spatial_rep.q_theta = (np.tensordot(spatial_rep.sphrharm_df1_mat,self.dr_pol_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.tor_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis]
        spatial_rep.q_phi = (-np.tensordot(spatial_rep.sphrharm_df1_mat,self.tor_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.dr_pol_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis]

        
    

    
    def calc_vel_grad(self,spatial_rep):
        
        if not hasattr(spatial_rep,'q_r'):
            
            self.calc_vel_field(spatial_rep)
        
        spatial_rep.dr_q_r = np.tensordot(spatial_rep.sphrharm_eval_mat,self.dr_pol_arr*(spatial_rep.l_arr*(spatial_rep.l_arr+1))[:,np.newaxis],axes=1).T/self.mb.r_grid[:,np.newaxis]**2 - 2 *spatial_rep.q_r/self.mb.r_grid[:,np.newaxis]
        spatial_rep.dr_q_theta = (np.tensordot(spatial_rep.sphrharm_df1_mat,self.dr2_pol_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.dr_tor_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis] - spatial_rep.q_theta/self.mb.r_grid[:,np.newaxis]
        spatial_rep.dr_q_phi = (-np.tensordot(spatial_rep.sphrharm_df1_mat,self.dr_tor_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.dr2_pol_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis] - spatial_rep.q_phi/self.mb.r_grid[:,np.newaxis]

        spatial_rep.ptheta_q_r =  np.tensordot(spatial_rep.sphrharm_df1_mat,self.pol_arr*(spatial_rep.l_arr*(spatial_rep.l_arr+1))[:,np.newaxis],axes=1).T/self.mb.r_grid[:,np.newaxis]**2
        spatial_rep.ptheta_q_theta = (np.tensordot(spatial_rep.sphrharm_df2_mat,self.dr_pol_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_df1_mat,self.tor_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:]-1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.tor_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:]**2*np.cos(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis]
        spatial_rep.ptheta_q_phi = (-np.tensordot(spatial_rep.sphrharm_df2_mat,self.tor_arr,axes=1).T+1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_df1_mat,self.dr_pol_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:]-1j*self.mb.m*np.tensordot(spatial_rep.sphrharm_eval_mat,self.dr_pol_arr,axes=1).T/np.sin(spatial_rep.theta_grid)[np.newaxis,:]**2*np.cos(spatial_rep.theta_grid)[np.newaxis,:])/self.mb.r_grid[:,np.newaxis]

        spatial_rep.dtheta_q_r = (spatial_rep.ptheta_q_r-spatial_rep.q_theta)/self.mb.r_grid[:,np.newaxis]
        spatial_rep.dphi_q_r = (1j*self.mb.m*spatial_rep.q_r/np.sin(spatial_rep.theta_grid)[np.newaxis,:]-spatial_rep.q_phi)/self.mb.r_grid[:,np.newaxis]

        spatial_rep.dtheta_q_theta = (spatial_rep.ptheta_q_theta+spatial_rep.q_r)/self.mb.r_grid[:,np.newaxis]
        spatial_rep.dphi_q_theta = (1j*self.mb.m*spatial_rep.q_theta-np.cos(spatial_rep.theta_grid)[np.newaxis,:]*spatial_rep.q_phi)/self.mb.r_grid[:,np.newaxis]/np.sin(spatial_rep.theta_grid)[np.newaxis,:]

        spatial_rep.dtheta_q_phi = spatial_rep.ptheta_q_phi/self.mb.r_grid[:,np.newaxis]
        spatial_rep.dphi_q_phi = (1j*self.mb.m*spatial_rep.q_phi+np.cos(spatial_rep.theta_grid)[np.newaxis,:]*spatial_rep.q_theta + np.sin(spatial_rep.theta_grid)[np.newaxis,:]*spatial_rep.q_r) /self.mb.r_grid[:,np.newaxis]/np.sin(spatial_rep.theta_grid)[np.newaxis,:]


                
    
class Spatial_representation:
    
    def __init__(self,theta_grid,soln):
        
        self.s = soln
        self.n_theta = len(theta_grid)
        self.theta_grid = theta_grid

        self.gen_theta_matrices()


    def gen_theta_matrices(self):
        
        self.sphrharm_eval_mat = np.zeros((self.n_theta,self.s.mb.n_l))
        self.sphrharm_df1_mat = np.zeros((self.n_theta,self.s.mb.n_l))


        s_max = self.s.mb.l_max+1

        for l in range(self.s.mb.l_min,s_max):
            
            i = l - self.s.mb.l_min
            self.sphrharm_eval_mat[:,i] = np.real(sphrharm(l,self.s.mb.m,self.theta_grid,0))

        self.sphrharm_df1_mat[:,-1] = (self.s.mb.l_max+1) * self.s.mb.c_l[-1] * np.real(sphrharm(self.s.mb.l_max-1,self.s.mb.m,self.theta_grid,0))
        for l in range(self.s.mb.l_min,s_max):
            
            i = l - self.s.mb.l_min
            self.sphrharm_df1_mat[:,i] = np.real(l * self.s.mb.c_l[i+1] * sphrharm(l+1,self.s.mb.m,self.theta_grid,0) - (l+1)*self.s.mb.c_l[i] * sphrharm(l-1,self.s.mb.m,self.theta_grid,0))

        self.sphrharm_df1_mat *= 1/np.sin(self.theta_grid)[:,np.newaxis]

        self.l_arr = np.linspace(self.s.mb.l_min,self.s.mb.l_max,self.s.mb.n_l)

        self.sphrharm_df2_mat = -self.sphrharm_df1_mat * np.cos(self.theta_grid[:,np.newaxis])/np.sin(self.theta_grid[:,np.newaxis])
        self.sphrharm_df2_mat +=  ((self.s.mb.m/np.sin(self.theta_grid[:,np.newaxis]))**2-self.l_arr*(self.l_arr+1)[np.newaxis,:]) * self.sphrharm_eval_mat
    
    def calc_bulk_dissipation(self):
        
        
        if not hasattr(self, 'dq_r_r'):
            
            self.s.calc_vel_grad(self)
        
        stress_rr = 2*self.s.ek*self.dr_q_r
        stress_rtheta = (self.dr_q_theta + self.dtheta_q_r)*self.s.ek
        stress_rphi =(self.dr_q_phi + self.dphi_q_r)*self.s.ek

        stress_thetatheta = 2*self.dtheta_q_theta*self.s.ek
        stress_thetaphi = (self.dphi_q_theta + self.dtheta_q_phi)*self.s.ek
        stress_phiphi = 2*self.dphi_q_phi*self.s.ek

        self.dissipation = self.dr_q_r * np.conjugate(stress_rr)
        self.dissipation += self.dtheta_q_r * np.conjugate(stress_rtheta)
        self.dissipation += self.dphi_q_r * np.conjugate(stress_rphi)

        self.dissipation += self.dr_q_theta * np.conjugate(stress_rtheta)
        self.dissipation += self.dtheta_q_theta * np.conjugate(stress_thetatheta)
        self.dissipation += self.dphi_q_theta * np.conjugate(stress_thetaphi)

        self.dissipation += self.dr_q_phi * np.conjugate(stress_rphi)
        self.dissipation += self.dtheta_q_phi * np.conjugate(stress_thetaphi)
        self.dissipation += self.dphi_q_phi * np.conjugate(stress_phiphi)

        self.dissipation = 1/2 * np.real(self.dissipation)


        self.total_dissipation = 2*np.pi * np.trapz(np.sin(self.theta_grid) * np.trapz(self.s.mb.r_grid[::-1,np.newaxis]**2*self.dissipation[::-1,:],x=self.s.mb.r_grid[::-1],axis=0),x=self.theta_grid,axis=0)

    def calc_total_kin(self):
        
        if not hasattr(self,'q_r'):
            
            self.s.calc_vel_field(self)
        
        self.kinetic_energy = 1/4 * (np.abs(self.q_r)**2+np.abs(self.q_theta)**2+np.abs(self.q_phi)**2)
        
        self.total_kinetic_energy = 2*np.pi * np.trapz(np.sin(self.theta_grid) * np.trapz(self.s.mb.r_grid[::-1,np.newaxis]**2*self.kinetic_energy[::-1,:],x=self.s.mb.r_grid[::-1],axis=0),x=self.theta_grid,axis=0)
    
    def calc_surface_power(self,full_calc=False):
        
        if not hasattr(self, 'dq_r_r'):
            
            self.s.calc_vel_grad(self)
        
        tau_phi_r =  self.dphi_q_r + self.dr_q_phi 
        tau_theta_r = self.dtheta_q_r + self.dr_q_theta
        
        
        self.power_cmb = 2*np.pi * self.s.ek* self.s.mb.r_end**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.real(self.q_phi[0,:]*np.conjugate(tau_phi_r[0,:])+self.q_theta[0,:]*np.conjugate(tau_theta_r[0,:])),x=self.theta_grid,axis=0)
        self.power_icb = 2*np.pi * self.s.ek* self.s.mb.r_start**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.real(self.q_phi[-1,:]*np.conjugate(tau_phi_r[-1,:])+self.q_theta[-1,:]*np.conjugate(tau_theta_r[-1,:])),x=self.theta_grid,axis=0)
        
        
        if full_calc:
            
            self.power_cmb_real_coeff = 2*np.pi * self.s.ek* self.s.mb.r_end**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.real(self.q_phi[0,:]*tau_phi_r[0,:]+self.q_theta[0,:]*tau_theta_r[0,:]),x=self.theta_grid,axis=0)
            self.power_cmb_imag_coeff = 2*np.pi * self.s.ek* self.s.mb.r_end**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.imag(self.q_phi[0,:]*tau_phi_r[0,:]+self.q_theta[0,:]*tau_theta_r[0,:]),x=self.theta_grid,axis=0)
            self.power_icb_real_coeff = 2*np.pi * self.s.ek* self.s.mb.r_start**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.real(self.q_phi[-1,:]*tau_phi_r[-1,:]+self.q_theta[-1,:]*tau_theta_r[-1,:]),x=self.theta_grid,axis=0)
            self.power_icb_imag_coeff = 2*np.pi * self.s.ek* self.s.mb.r_start**2* np.trapz(np.sin(self.theta_grid) * 0.5*np.imag(self.q_phi[-1,:]*tau_phi_r[-1,:]+self.q_theta[-1,:]*tau_theta_r[-1,:]),x=self.theta_grid,axis=0)
            
    def calc_advection(self):
        if not hasattr(self, 'dq_r_r'):
            
            self.s.calc_vel_grad(self)
        self.mean_adv_r = 1/2 * np.real(self.q_r*np.conjugate(self.dr_q_r)+self.q_theta*np.conjugate(self.dtheta_q_r)+self.q_phi*np.conjugate(self.dphi_q_r))
        self.mean_adv_theta = 1/2 * np.real(self.q_r*np.conjugate(self.dr_q_theta)+self.q_theta*np.conjugate(self.dtheta_q_theta)+self.q_phi*np.conjugate(self.dphi_q_theta))
        self.mean_adv_phi = 1/2 * np.real(self.q_r*np.conjugate(self.dr_q_phi)+self.q_theta*np.conjugate(self.dtheta_q_phi)+self.q_phi*np.conjugate(self.dphi_q_phi))


class PDE_matrix_frame:
    
    def __init__(self,matrix,matrix_builder,ek,for_freq):
        
        self.ek = ek
        self.for_freq = for_freq
        self.mb = matrix_builder
        self.matrix = matrix
        
    def calc_LU(self):
        
        self.LU = spla.splu(self.matrix)
        
    def solve_sys(self,rhs):
        
        if not hasattr(self, 'LU'):
            self.calc_LU()
        
        return Soln_forced(self.LU.solve(rhs),self.mb,self.ek,self.for_freq)

class Boundary_rhs_builder:
    
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
                

   





