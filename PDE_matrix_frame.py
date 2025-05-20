import numpy as np
import scipy.sparse.linalg as spla
from utility_functions import sphrharm
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
    

class Soln_forced:
    
    def __init__(self,soln,mat_builder,ek,for_freq):
        
        self.mb = mat_builder
        self.soln = soln
        self.ek = ek
        self.for_freq = for_freq
        

    def process_soln(self,odd_flag):
        

        tor_arr = np.zeros((self.mb.n_l,self.mb.n_rad_max+1),dtype=complex)
        dr_tor_arr = np.zeros_like(tor_arr)
        pol_arr = np.zeros((self.mb.n_l,self.mb.n_rad_max+1),dtype=complex)
        dr_pol_arr = np.zeros((self.mb.n_l,self.mb.n_rad_max+1),dtype=complex)
        dr2_pol_arr = np.zeros_like(tor_arr)
        
        if self.mb.rad_ratio > 0.0:
            if odd_flag == 'tor':
                
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
        elif self.mb.rad_ratio == 0.0:
            
            if odd_flag == 'tor':
                
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat_even @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat_even @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat_odd @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat_odd @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat_odd @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
            if odd_flag == 'pol':
                
                for l in self.mb.l_even_full:
                    
                    i = l-self.mb.l_min
                    tor_arr[i,:] = self.mb.eval_mat_odd @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_tor_arr[i,:] = self.mb.df1_mat_odd @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    
                for l in self.mb.l_odd_full:
                    
                    i = l-self.mb.l_min
                    pol_arr[i,:] = self.mb.eval_mat_even @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr_pol_arr[i,:] = self.mb.df1_mat_even @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
                    dr2_pol_arr[i,:] = self.mb.df2_mat_even @ self.soln[i*(self.mb.n_rad_max+1):(i+1)*(self.mb.n_rad_max+1)]
            
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

