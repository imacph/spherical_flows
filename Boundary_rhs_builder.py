import numpy as np
from utility_functions import l_setup

class Boundary_rhs_builder:
    
    def __init__(self,n_rad_max,rad_ratio,m,l_max):
        
        self.n_rad_max = n_rad_max
        self.rad_ratio = rad_ratio
        self.m = m
        self.l_max = l_max
        
    
        self.l_min,self.n_l,self.l_odd,self.l_even,self.l_even_full,self.l_odd_full = l_setup(self.m,self.l_max)
        
        
    def gen_rhs(self,bc_list,odd_flag):
        
        self.gen_bc_arrs(bc_list)
        
        self.rhs_tor_odd = np.zeros(self.n_l*(self.n_rad_max+1),dtype=complex)
        self.rhs_tor_even = np.zeros_like(self.rhs_tor_odd)
        
    
        for l in self.l_odd_full:
            
            i = l - self.l_min
            self.rhs_tor_odd[i*(self.n_rad_max+1)] = self.tor_t[i]
            self.rhs_tor_odd[(i+1)*(self.n_rad_max+1)-1] = self.tor_b[i]
            
            self.rhs_tor_even[i*(self.n_rad_max+1)] = self.pol_t[i]
            self.rhs_tor_even[i*(self.n_rad_max+1)+1] = self.dr_pol_t[i]
            self.rhs_tor_even[(i+1)*(self.n_rad_max+1)-1] = self.pol_b[i]
            self.rhs_tor_even[(i+1)*(self.n_rad_max+1)-2] = self.dr_pol_b[i]
            
        for l in self.l_even_full:
            
            i = l - self.l_min
            self.rhs_tor_even[i*(self.n_rad_max+1)] = self.tor_t[i]
            self.rhs_tor_even[(i+1)*(self.n_rad_max+1)-1] = self.tor_b[i]
            
            self.rhs_tor_odd[i*(self.n_rad_max+1)] = self.pol_t[i]
            self.rhs_tor_odd[i*(self.n_rad_max+1)+1] = self.dr_pol_t[i]
            self.rhs_tor_odd[(i+1)*(self.n_rad_max+1)-1] = self.pol_b[i]
            self.rhs_tor_odd[(i+1)*(self.n_rad_max+1)-2] = self.dr_pol_b[i]
    
        if odd_flag == 'tor':
            
            self.rhs = self.rhs_tor_odd
        
        elif odd_flag == 'pol':
        
            self.rhs = self.rhs_tor_even
        

    def gen_bc_arrs(self,bc_list):
        
        # bc_list = list of lists with each sublist formatted as: ['tor/pol/dr_pol','t/b','l',value]
        
        self.pol_t = np.zeros(self.n_l*(self.n_rad_max+1),dtype=complex)
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
                

   