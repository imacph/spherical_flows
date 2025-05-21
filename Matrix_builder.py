import numpy as np
from scipy import sparse as sp
from utility_functions import l_setup   


class Matrix_builder:
    
    def __init__(self,n_rad_max,rad_ratio,m,l_max,radial_endpoint_inclusive=True,stress_free_icb = False,stress_free_cmb=False,
                    radial_method="chebyshev",radial_grid=None):
        
        self.n_rad_max = n_rad_max
        self.rad_ratio = rad_ratio
        self.m = m
        self.l_max = l_max
        
        self.r_end = 1/(1-rad_ratio)
        self.r_start = self.r_end * rad_ratio
        
        self.r_fac = 2/(self.r_end-self.r_start)
        
        self.stress_free_icb = stress_free_icb
        self.stress_free_cmb = stress_free_cmb

        self.radial_endpoint_inclusive = radial_endpoint_inclusive
        self.radial_method = radial_method

        if radial_endpoint_inclusive:

            if self.rad_ratio > 0.0:

                
                self.x_grid = np.cos(np.linspace(0,n_rad_max,n_rad_max+1)*np.pi/n_rad_max)
                self.r_grid = self.x_grid * (self.r_end-self.r_start)/2 + (self.r_end+self.r_start)/2
                self.eval_mat,self.df1_mat,self.df2_mat,self.df4_mat = self.gen_deriv_mats()
            
            elif self.rad_ratio == 0.0:

                
                self.x_grid = np.cos(np.linspace(0,n_rad_max,n_rad_max+1)*np.pi/(2*n_rad_max+1))
                self.r_grid = self.x_grid
                self.eval_mat_even,self.eval_mat_odd,self.df1_mat_even,self.df1_mat_odd,self.df2_mat_even,self.df2_mat_odd,self.df4_mat_even,self.df4_mat_odd = self.gen_deriv_mats_fs()
            
        else:
            # placeholder for the case where the radial endpoints are not included
            # this will be used for example if we want to solve eigenvalue problems
            pass


        self.l_min,self.n_l,self.l_odd,self.l_even,self.l_even_full,self.l_odd_full = l_setup(self.m,self.l_max)
        
        
        self.orr_mat = sp.csr_matrix(np.diag(1/self.r_grid),dtype=complex)
        self.orr_sqr_mat = self.orr_mat * self.orr_mat
            
    
        self.c_l = np.zeros(self.n_l+1)
        
        for l in range(self.l_min,self.l_max+2):
            
            i = l - self.l_min
            
            self.c_l[i] = np.sqrt((l-self.m)*(l+self.m)/(2*l+1)/(2*l-1))
        
    
    def gen_deriv_mats_fs(self):   

        grid_diag = sp.csr_matrix(np.diag(self.x_grid),dtype=complex)
        grid_sqr_diag = grid_diag * grid_diag
        
        if self.radial_method == 'chebyshev':
            df1_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df2_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df3_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df4_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            eval_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            
            df1_mat_odd = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df2_mat_odd = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df4_mat_odd = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            eval_mat_odd = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            
            df1_mat_even = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df2_mat_even = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df4_mat_even = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            eval_mat_even = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            

            for i in range(self.n_rad_max+1):

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
        
        elif self.radial_method == 'finite_difference':
            
            eval_mat,df1_mat,df2_mat,df4_mat = self.gen_deriv_mats()
            df3_mat = df2_mat @ df1_mat

            eval_mat_odd = grid_sqr_diag @ eval_mat
            df1_mat_odd = grid_sqr_diag @ df1_mat + 2*grid_diag @ eval_mat
            df2_mat_odd = grid_sqr_diag @ df2_mat + 4*grid_diag @ df1_mat + 2 * eval_mat
            df4_mat_odd = grid_sqr_diag @ df4_mat + 8 * grid_diag @ df3_mat + 12 * df2_mat

            eval_mat_even = grid_diag @ eval_mat
            df1_mat_even = grid_diag @ df1_mat + eval_mat
            df2_mat_even = grid_diag @ df2_mat + 2*df1_mat
            df4_mat_even = grid_diag @ df4_mat + 4 *df3_mat

            return eval_mat_even,eval_mat_odd,df1_mat_even,df1_mat_odd,df2_mat_even,df2_mat_odd,df4_mat_even,df4_mat_odd

            


    def gen_deriv_mats(self):
        
        if self.radial_method == 'chebyshev':

            df1_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df2_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df4_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            eval_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            
            for n in range(self.n_rad_max+1):
                
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
        
        elif self.radial_method == 'finite_difference':

            eval_mat = sp.identity(self.n_rad_max+1,dtype=complex)
            df1_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))
            df2_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1))

            df1_mat[0,0] = -1/(self.r_grid[1]-self.r_grid[0])
            df1_mat[0,1] = 1/(self.r_grid[1]-self.r_grid[0])

            df1_mat[-1,-1] = 1/(self.r_grid[-1]-self.r_grid[-2])
            df1_mat[-1,-2] = -1/(self.r_grid[-1]-self.r_grid[-2])

            df2_mat[0,0] = 2/(self.r_grid[2]-self.r_grid[0])/(self.r_grid[1]-self.r_grid[0])
            df2_mat[0,1] = -2/(self.r_grid[1]-self.r_grid[0])/(self.r_grid[2]-self.r_grid[1])
            df2_mat[0,2] = 2/(self.r_grid[2]-self.r_grid[0])/(self.r_grid[2]-self.r_grid[1])

            df2_mat[-1,-1] = 2/(self.r_grid[-3]-self.r_grid[-1])/(self.r_grid[-2]-self.r_grid[-1])
            df2_mat[-1,-2] = -2/(self.r_grid[-2]-self.r_grid[-1])/(self.r_grid[-3]-self.r_grid[-2])
            df2_mat[-1,-3] = 2/(self.r_grid[-3]-self.r_grid[-1])/(self.r_grid[-3]-self.r_grid[-2])
            
            for i in range(1,self.n_rad_max):

                df1_mat[i,i-1] = -1/(self.r_grid[i+1]-self.r_grid[i-1])
                df1_mat[i,i+1] = 1/(self.r_grid[i+1]-self.r_grid[i-1])

                df2_mat[i,i-1] = 2/(self.r_grid[i]-self.r_grid[i-1])/(self.r_grid[i+1]-self.r_grid[i-1])
                df2_mat[i,i] = -2/(self.r_grid[i+1]-self.r_grid[i])/(self.r_grid[i]-self.r_grid[i-1])
                df2_mat[i,i+1] = 2/(self.r_grid[i+1]-self.r_grid[i])/(self.r_grid[i+1]-self.r_grid[i-1])

            return eval_mat,df1_mat,df2_mat,df2_mat@df2_mat



    def gen_block_row(self,field,l,for_freq,ek):
        
        lp = l*(l+1)
        lp2 = lp*lp
        
        lower_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1),dtype=complex)
        upper_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1),dtype=complex)
        diag_mat = sp.csr_matrix((self.n_rad_max+1,self.n_rad_max+1),dtype=complex)
        
        if self.rad_ratio > 0.0:
        
            if field == 'toroidal':
                
                diag_mat[1:-1,:] += (1j * (lp*for_freq-2*self.m) * self.eval_mat + ek*lp2*self.orr_sqr_mat*self.eval_mat)[1:-1,:]
                diag_mat[1:-1,:] += -lp * ek * self.df2_mat[1:-1,:]
                
                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat[0,:]
                    diag_mat[-1,:] = self.eval_mat[-1,:]
                    #diag_mat[-1,:] = ( self.df1_mat-2*self.orr_mat * self.eval_mat)[-1,:]

                elif self.radial_method == 'finite_difference':

                    diag_mat[0,0] = 1
                    diag_mat[-1,-1] = 1

                
                
                
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
                

                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat[0,:]
                    diag_mat[-1,:] = self.eval_mat[-1,:]
                    diag_mat[1,:] = self.df1_mat[0,:]
                    diag_mat[-2,:] = self.df1_mat[-1,:]
                    #diag_mat[-2,:] = (self.df2_mat-2*self.orr_mat * self.df1_mat)[-1,:]
                elif self.radial_method == 'finite_difference':

                    diag_mat[0,0] = 1
                    diag_mat[1,0] = 1
                    diag_mat[1,1] = self.r_grid[1] - self.r_grid[0]

                    diag_mat[-1,-1] = 1
                    diag_mat[-2,-1] = 1
                    diag_mat[-2,-2] = self.r_grid[-2] - self.r_grid[-1]
                
                
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
                
                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat_odd[0,:]
                elif self.radial_method == 'finite_difference':
                    diag_mat[0,0] = 1

                
                upper_fac = 2*l*(l+2) * np.sqrt((l+1+self.m)*(l+1-self.m)/(2*l+1)/(2*l+3))
                
                upper_mat[1:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_even)[1:]
                upper_mat[1:] += -upper_fac * self.df1_mat_even[1:]
    
    
                lower_fac = 2*(l-1)*(l+1) * np.sqrt((l+self.m)*(l-self.m)/(2*l-1)/(2*l+1))
                
                lower_mat[1:] += lower_fac * l*(self.orr_mat*self.eval_mat_even)[1:]
                lower_mat[1:] += -lower_fac * self.df1_mat_even[1:]
                
            elif field == 'toroidal':
                
                diag_mat[1:] += (1j * (lp*for_freq-2*self.m) * self.eval_mat_even + ek*lp2*self.orr_sqr_mat*self.eval_mat_even)[1:]
                diag_mat[1:] += -lp * ek * self.df2_mat_even[1:]
                
                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat_even[0,:]
                elif self.radial_method == 'finite_difference':
                    diag_mat[0,0] = 1

                
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
                
                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat_odd[0,:]
                    diag_mat[1,:] = self.df1_mat_odd[0,:]
                elif self.radial_method == 'finite_difference':
                    diag_mat[0,0] = 1
                    diag_mat[1,0] = 1
                    diag_mat[1,1] = self.r_grid[1] - self.r_grid[0]

                    
                


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
                
                if self.radial_method == 'chebyshev':
                    diag_mat[0,:] = self.eval_mat_even[0,:]
                    diag_mat[1,:] = self.df1_mat_even[0,:]
                elif self.radial_method == 'finite_difference':
                    diag_mat[0,0] = 1
                    diag_mat[1,0] = 1
                    diag_mat[1,1] = self.r_grid[1] - self.r_grid[0]


                upper_fac = 2*l*(l+2) * self.c_l[l+1-self.l_min]
                
                upper_mat[2:] += -upper_fac * (l+1)*(self.orr_mat*self.eval_mat_odd)[2:]
                upper_mat[2:] += -upper_fac * self.df1_mat_odd[2:]
                
            
                lower_fac = 2*(l-1)*(l+1)* self.c_l[l-self.l_min]
                
                lower_mat[2:] += lower_fac * l*(self.orr_mat*self.eval_mat_odd)[2:]
                lower_mat[2:] += -lower_fac * self.df1_mat_odd[2:]
            
        return diag_mat,lower_mat,upper_mat

    
    
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