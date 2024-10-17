import numpy as np
from sphrflow_main import LN,sphrharm
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from tqdm import tqdm

def Ekman_disp(freq,eps,E,eta):
    
    return 2*np.pi/15/np.sqrt(2) * np.abs(eps)**2 * E**(1/2)/(1-eta)**4 * ((2-freq)**(5/2)+(2+freq)**(5/2)-1/7*((2-freq)**(7/2)+(2+freq)**(7/2)))



N = 100
l_max = 80
N = 240
l_max = 200
rad_ratio = 0.
m = 0
ek = 10**(-8)

LN_case = LN(N,rad_ratio,m,l_max,eigen_flag=0)




n_theta = LN_case.n_l
theta_min = 0
theta_max = np.pi
theta_grid = 0.5 * (np.cos(np.linspace(1,n_theta,n_theta)*np.pi/(n_theta+1))[::-1] * (theta_max-theta_min) + (theta_max+theta_min))

sphrharm_eval_mat = np.zeros((LN_case.n_l,LN_case.n_l))
sphrharm_df1_mat = np.zeros((LN_case.n_l,LN_case.n_l))

for l in range(LN_case.l_min,LN_case.l_max+1):
    
    i = l - LN_case.l_min
    sphrharm_eval_mat[:,i] = np.real(sphrharm(l,LN_case.m,theta_grid,0))

sphrharm_df1_mat[:,-1] = (LN_case.l_max+1) * LN_case.c_l[-1] * np.real(sphrharm(LN_case.l_max-1,LN_case.m,theta_grid,0))
for l in range(LN_case.l_min,LN_case.l_max):
    
    i = l - LN_case.l_min
    sphrharm_df1_mat[:,i] = np.real(l * LN_case.c_l[i+1] * sphrharm(l+1,LN_case.m,theta_grid,0) - (l+1)*LN_case.c_l[i] * sphrharm(l-1,LN_case.m,theta_grid,0))

sphrharm_df1_mat *= 1/np.sin(theta_grid)[:,np.newaxis]

l_arr = np.linspace(LN_case.l_min,LN_case.l_max,LN_case.n_l)

sphrharm_df2_mat = -sphrharm_df1_mat * np.cos(theta_grid[:,np.newaxis])/np.sin(theta_grid[:,np.newaxis])
sphrharm_df2_mat +=  ((LN_case.m/np.sin(theta_grid[:,np.newaxis]))**2-l_arr*(l_arr+1)[np.newaxis,:]) * sphrharm_eval_mat



n_it =3000
step = 4*(ek)**(0.23)/200
step = 3.33333333e-4
disp_arr = np.zeros(n_it)
surf_arr = np.zeros_like(disp_arr)
kin_arr = np.zeros_like(disp_arr)
#kin_spec = np.zeros((n_it,LN_case.n_l))
for k in tqdm(range(1,n_it+1)):
    soln = np.loadtxt('C:/Users/ianma/Desktop/solns/soln_{:d}.txt'.format(k),dtype=complex)
    
    tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr = LN_case.process_soln('tor',soln)
    '''
    for i in range(LN_case.n_l):
        
        
        sphrharm_eval_single = np.zeros_like(sphrharm_eval_mat)
        sphrharm_df_single  = np.zeros_like(sphrharm_eval_mat) 
        sphrharm_eval_single[:,i] = sphrharm_eval_mat[:,i]
        sphrharm_df_single[:,i] = sphrharm_df1_mat[:,i]
        
        q_r = np.tensordot(sphrharm_eval_single,pol_arr*(l_arr*(l_arr+1))[:,np.newaxis],axes=1).T/LN_case.r_grid[:,np.newaxis]**2
        q_theta = (np.tensordot(sphrharm_df_single,dr_pol_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_single,tor_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
        q_phi = (-np.tensordot(sphrharm_df_single,tor_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_single,dr_pol_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
        
        
        en = 1/4 * (np.abs(q_r)+np.abs(q_theta)+np.abs(q_phi))
        
        kin_spec[k-1,i] = 2*np.pi*np.trapz(np.sin(theta_grid)*np.trapz(en*LN_case.r_grid[::-1][:,np.newaxis]**2,x=LN_case.r_grid[::-1],axis=0),x=theta_grid)
    '''
    
    q_r = np.tensordot(sphrharm_eval_mat,pol_arr*(l_arr*(l_arr+1))[:,np.newaxis],axes=1).T/LN_case.r_grid[:,np.newaxis]**2
    q_theta = (np.tensordot(sphrharm_df1_mat,dr_pol_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_mat,tor_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
    q_phi = (-np.tensordot(sphrharm_df1_mat,tor_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_mat,dr_pol_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
    
    dr_q_r = np.tensordot(sphrharm_eval_mat,dr_pol_arr*(l_arr*(l_arr+1))[:,np.newaxis],axes=1).T/LN_case.r_grid[:,np.newaxis]**2 - 2 *q_r/LN_case.r_grid[:,np.newaxis]
    dr_q_theta = (np.tensordot(sphrharm_df1_mat,dr2_pol_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_mat,dr_tor_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis] - q_theta/LN_case.r_grid[:,np.newaxis]
    dr_q_phi = (-np.tensordot(sphrharm_df1_mat,dr_tor_arr,axes=1).T+1j*m*np.tensordot(sphrharm_eval_mat,dr2_pol_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis] - q_phi/LN_case.r_grid[:,np.newaxis]
    
    ptheta_q_r =  np.tensordot(sphrharm_df1_mat,pol_arr*(l_arr*(l_arr+1))[:,np.newaxis],axes=1).T/LN_case.r_grid[:,np.newaxis]**2
    ptheta_q_theta = (np.tensordot(sphrharm_df2_mat,dr_pol_arr,axes=1).T+1j*m*np.tensordot(sphrharm_df1_mat,tor_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:]-1j*m*np.tensordot(sphrharm_eval_mat,tor_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:]**2*np.cos(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
    ptheta_q_phi = (-np.tensordot(sphrharm_df2_mat,tor_arr,axes=1).T+1j*m*np.tensordot(sphrharm_df1_mat,dr_pol_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:]-1j*m*np.tensordot(sphrharm_eval_mat,dr_pol_arr,axes=1).T/np.sin(theta_grid)[np.newaxis,:]**2*np.cos(theta_grid)[np.newaxis,:])/LN_case.r_grid[:,np.newaxis]
    
    dtheta_q_r = (ptheta_q_r-q_theta)/LN_case.r_grid[:,np.newaxis]
    dphi_q_r = (1j*m*q_r/np.sin(theta_grid)[np.newaxis,:]-q_phi)/LN_case.r_grid[:,np.newaxis]
    
    dtheta_q_theta = (ptheta_q_theta+q_r)/LN_case.r_grid[:,np.newaxis]
    dphi_q_theta = (1j*LN_case.m*q_theta-np.cos(theta_grid)[np.newaxis,:]*q_phi)/LN_case.r_grid[:,np.newaxis]/np.sin(theta_grid)[np.newaxis,:]
    
    dtheta_q_phi = ptheta_q_phi/LN_case.r_grid[:,np.newaxis]
    dphi_q_phi = (1j*LN_case.m*q_phi+np.cos(theta_grid)[np.newaxis,:]*q_theta + np.sin(theta_grid)[np.newaxis,:]*q_r) /LN_case.r_grid[:,np.newaxis]/np.sin(theta_grid)[np.newaxis,:]
    
    stress_rr = 2*dr_q_r
    stress_rtheta = dr_q_theta + dtheta_q_r
    stress_rphi = dr_q_phi + dphi_q_r
    
    stress_thetatheta = 2*dtheta_q_theta
    stress_thetaphi = dphi_q_theta + dtheta_q_phi
    stress_phiphi = 2*dphi_q_phi
    
    disp = dr_q_r * np.conjugate(stress_rr)
    disp += dtheta_q_r * np.conjugate(stress_rtheta)
    disp += dphi_q_r * np.conjugate(stress_rphi)
    
    disp += dr_q_theta * np.conjugate(stress_rtheta)
    disp += dtheta_q_theta * np.conjugate(stress_thetatheta)
    disp += dphi_q_theta * np.conjugate(stress_thetaphi)
    
    disp += dr_q_phi * np.conjugate(stress_rphi)
    disp += dtheta_q_phi * np.conjugate(stress_thetaphi)
    disp += dphi_q_phi * np.conjugate(stress_phiphi)
    
    disp = 1/2 * np.real(disp)
    
    tot_disp = ek*2*np.pi * np.trapz(np.sin(theta_grid) * np.trapz(LN_case.r_grid[::-1,np.newaxis]**2*disp[::-1,:],x=LN_case.r_grid[::-1],axis=0),x=theta_grid,axis=0)
    
    tau_phi_r =  dphi_q_r + dr_q_phi 
    tau_theta_r = dtheta_q_r + dr_q_theta


    surf_power = 2*np.pi * ek* LN_case.r_end**2* np.trapz(np.sin(theta_grid) * 0.5*np.real(q_phi[0,:]*np.conjugate(tau_phi_r[0,:])+q_theta[0,:]*np.conjugate(tau_theta_r[0,:])),x=theta_grid,axis=0)

    disp_arr[k-1] = surf_power
    surf_arr[k-1] = surf_power
    
    kin = 1/4 * (np.abs(q_r)**2+np.abs(q_theta)**2+np.abs(q_phi)**2)
    
    kin_arr[k-1] = -2*np.pi*np.trapz(np.sin(theta_grid)*np.trapz(kin*LN_case.r_grid[:,np.newaxis]**2,x=LN_case.r_grid,axis=0),x=theta_grid,axis=0)


fig,ax = plt.subplots(1,1,figsize=(8,5),dpi=200)

freq0 = 1.315909340078351
freq0 = -1.4406
freq0 = -1.0614748821994373
#freq0 = -1.24538

freq0 = 2*np.sin(np.pi/6)
freqs = np.linspace(step,n_it*step,n_it)+freq0-2*ek**(0.23)
freqs = np.linspace(step,n_it*step,n_it)
#np.savetxt('revision_files/ek_65_pi6/freqs.txt',freqs)
#np.savetxt('revision_files/ek_65_pi6/disp.txt',disp_arr)
#np.savetxt('revision_files/ek_65_pi6/kin.txt',kin_arr)
np.savetxt('C:/Users/ianma/Desktop/solns/freqs.txt',freqs)
np.savetxt('C:/Users/ianma/Desktop/solns/disp.txt',disp_arr)
np.savetxt('C:/Users/ianma/Desktop/solns/kin.txt',kin_arr)

#np.savetxt('kin_spec_4_topo.txt',kin_spec)

ax.plot(freqs,disp_arr)
#ax.plot(freqs,disp_arr/Ekman_disp(freqs,1,ek,0))
#ax.plot(freqs,surf_arr/Ekman_disp(freqs,1,ek,0))
print(np.max(np.abs(disp_arr-surf_arr)/disp_arr))
#ax.plot(freqs,Ekman_disp(freqs,1,ek,0))

#ax.set_ylim(0.85,1.04)
ax.axvline(freq0)

#prelim = np.loadtxt('DNS_disp_E4.txt')

#ax.plot(prelim[:,0],prelim[:,1],ls='none',marker='s',mfc='none',ms=3,color='red',zorder=10)
