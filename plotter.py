import numpy as np
from scipy import sparse as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import lpmv
import scipy.sparse.linalg as spla
from sphrflow_main import LN,sphrharm,Ekman_disp


N = 200
l_max = 60
rad_ratio = 0.0
m = 0

for_freq = 1
ek = 1e-4
bc_list = [['tor','t',1,2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]


t0 = time()
LN_case = LN(N,rad_ratio,m,l_max,eigen_flag=0)


PDE_mat = LN_case.gen_PDE_matrix('tor',for_freq,ek)

LU = spla.splu(PDE_mat)

tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b = LN_case.gen_bc_arrs(bc_list)

rhs_tor_odd,rhs_tor_even = LN_case.gen_rhs(tor_t,tor_b,pol_t,pol_b,dr_pol_t,dr_pol_b)
soln = LU.solve(rhs_tor_odd)
tor_arr,dr_tor_arr,pol_arr,dr_pol_arr,dr2_pol_arr = LN_case.process_soln('tor',soln)
print(time()-t0)



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

print(surf_power/Ekman_disp(for_freq,1,ek,0))

print(np.abs(1-tot_disp/surf_power),np.abs(1-surf_power/tot_disp))




en = 1/4*(np.abs(q_r)**2+np.abs(q_theta)**2+np.abs(q_phi)**2)

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
    
    
    off=-7
    top = 0
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






