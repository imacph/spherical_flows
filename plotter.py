import numpy as np
from scipy import sparse as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import lpmv
import scipy.sparse.linalg as spla
from sphrflow_main import Matrix_builder_forced,Boundary_rhs_builder,Soln_forced,sphrharm,Spatial_representation,PDE_matrix_frame
import tracemalloc
from math import factorial

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
            
def C(m,k,i,j):
    
    return (-1)**(i+j)*double_fac(2*(m+k+i+j)-1)/2**(j+1)/double_fac(2*i-1)/fac(k-i-j)/fac(i)/fac(j)/fac(m+j)

def calc_q_r(m,k,freq,r,theta):
    
    sig = freq/2
    nr = len(r)
    ntheta = len(theta)
    orr = 1/r
    field = np.zeros((nr,ntheta),dtype=complex)
    
    for i in range(k+1):
        
        for j in range(k-i+1):
            
            field += C(m,k,i,j) * orr[:,np.newaxis] * (sig**2*(m+2*j)+m*sig-2*i*(1-sig**2)) * (r[:,np.newaxis]**(m+2*(i+j))*sig**(2*i-1)*(1-sig**2)**(j-1)*np.sin(theta[np.newaxis,:])**(m+2*j)*np.cos(theta[np.newaxis,:])**(2*i))
            
    return field*-1j/2
            
      
def calc_q_theta(m,k,freq,r,theta):
    
    sig = freq/2
    nr = len(r)
    ntheta = len(theta)
    orr = 1/r
    field = np.zeros((nr,ntheta),dtype=complex)
    
    for i in range(k+1):
        
        for j in range(k-i+1):
            
            field += C(m,k,i,j) * orr[:,np.newaxis] * ((sig**2*(m+2*j)+m*sig)*np.cos(theta[np.newaxis,:])**2+2*i*(1-sig**2)*np.sin(theta[np.newaxis,:])**2) * (r[:,np.newaxis]**(m+2*(i+j))*sig**(2*i-1)*(1-sig**2)**(j-1)*np.sin(theta[np.newaxis,:])**(m+2*j-1)*np.cos(theta[np.newaxis,:]) ** (2*i-1))
    return field*-1j/2
            
         
def calc_q_phi(m,k,freq,r,theta):
    
    sig = freq/2
    nr = len(r)
    ntheta = len(theta)
    orr = 1/r
    field = np.zeros((nr,ntheta),dtype=complex)
    
    for i in range(k+1):
        
        for j in range(k-i+1):
            
            field += C(m,k,i,j)*orr[:,np.newaxis]*((m+2*j)+m*sig) * (r[:,np.newaxis]**(m+2*(i+j))*sig**(2*i)*(1-sig**2)**(j-1)*np.sin(theta[np.newaxis,:])**(m+2*j-1)*np.cos(theta[np.newaxis,:])**(2*i))
            
    return field*1/2
            
             
def calc_norm(m,k,freq):
    
    sig = freq/2
    out = 0
    for i in range(k+1):
        
        for j in range(k-i+1):
            
            for q in range(k+1):
                
                for l in range(k-q+1):
                    
                    
                    out += (-1)**(i+j+q+l) * np.pi * fac(m+j+l-1)/2**(3-m) /double_fac(2*(m+i+j+q+l)+1) * (8*q*i*double_fac(2*i+2*q-3)*(m+j+l)/sig**2+((m*sig+m+2*j*sig)*(m*sig+m+2*l*sig)/(1-sig**2)**2+(m*sig+m+2*j)*(m*sig+m+2*l)/(1-sig**2)**2)* double_fac(2*i+2*q-1)) *sig**(2*(i+q))/double_fac(2*i-1) *(1-sig**2)**(j+l) * double_fac(2*(m+k+i+j)-1) * double_fac(2*(m+k+q+l)-1) / double_fac(2*q-1) / fac(k-i-j) / fac(k-q-l) / fac(i) / fac(q) / fac(j) /fac(l) / fac(j+m) / fac(m+l)
                    

    return out

def Ekman_disp(freq,eps,E,eta):
    
    return 2*np.pi/15/np.sqrt(2) * np.abs(eps)**2 * E**(1/2)/(1-eta)**4 * ((2-freq)**(5/2)+(2+freq)**(5/2)-1/7*((2-freq)**(7/2)+(2+freq)**(7/2)))

 
def Ekman_kin(freq,eps,E,eta):
    
    return np.pi/3/np.sqrt(2) * eps**2*E**(1/2)/(1-eta)**4 * ((2-freq)**(3/2)+(2+freq)**(3/2)-1/5*((2-freq)**(5/2)+(2+freq)**(5/2)))
 
 
'resolution and symmetry parameters'
N =60 # number of radial grid points in soln.
l_max =50 # maximum spherical harmonic degree in soln.
rad_ratio = 0.6 # spherical shell aspect ratio
m = 0 # azimuthal symmetry order

'libration parameters'
ek = 10**-4 # ekman number
Re = 300 # reynolds number
for_freq = 1. # forcing frequency

# calculates the appropriate libration amplitude
eps = Re*np.sqrt(ek)*(1-rad_ratio)

# information about BCs to pass to solver
bc_list = [['tor','t',1,eps*2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]

'matrix construction and inverse problem solution'
t0 = time() 

# building the PDE matrix
matrix_builder = Matrix_builder_forced(N,rad_ratio,m,l_max)
PDE_mat = PDE_matrix_frame(matrix_builder.gen_PDE_matrix('tor',for_freq,ek),matrix_builder,ek,for_freq)

# building the libration forcing RHS
rhs_builder = Boundary_rhs_builder(N,rad_ratio,m,l_max)
rhs_builder.gen_rhs(bc_list,'tor')

# solving inverse problem and formatting solution arrays
PDE_soln = PDE_mat.solve_sys(rhs_builder.rhs)
PDE_soln.process_soln('tor')

# the time to complete the solution
sol_time = time()-t0
print(sol_time)

# calculating theta_grid for spatial representation of solution
n_theta = 4*PDE_soln.mb.n_l
theta_min = 0
theta_max = np.pi
theta_grid = 0.5 * (np.cos(np.linspace(1,n_theta,n_theta)*np.pi/(n_theta+1))[::-1] * (theta_max-theta_min) + (theta_max+theta_min))

spat_rep = Spatial_representation(theta_grid,PDE_soln)

# calculating velocity field and its gradients
PDE_soln.calc_vel_field(spat_rep)
PDE_soln.calc_vel_grad(spat_rep)

'shortening names'
q_r,q_theta,q_phi = spat_rep.q_r,spat_rep.q_theta,spat_rep.q_phi

dr_q_r,dtheta_q_r,dphi_q_r = spat_rep.dr_q_r,spat_rep.dtheta_q_r,spat_rep.dphi_q_r 
dr_q_theta,dtheta_q_theta,dphi_q_theta = spat_rep.dr_q_theta,spat_rep.dtheta_q_theta,spat_rep.dphi_q_theta
dr_q_phi,dtheta_q_phi,dphi_q_phi = spat_rep.dr_q_phi,spat_rep.dtheta_q_phi,spat_rep.dphi_q_phi 

en = 1/4*(np.abs(q_r)**2+np.abs(q_theta)**2+np.abs(q_phi)**2)

tot_en = -2*np.pi*np.trapz(np.sin(theta_grid)*np.trapz(PDE_soln.mb.r_grid[:,np.newaxis]**2*en,x=PDE_soln.mb.r_grid,axis=0),x=theta_grid,axis=0)
print(tot_en*47**2)


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


tot_disp = ek*2*np.pi * np.trapz(np.sin(theta_grid) * np.trapz(PDE_soln.mb.r_grid[::-1,np.newaxis]**2*disp[::-1,:],x=PDE_soln.mb.r_grid[::-1],axis=0),x=theta_grid,axis=0)




tau_phi_r =  dphi_q_r + dr_q_phi 
tau_theta_r = dtheta_q_r + dr_q_theta


surf_power = 2*np.pi * ek* PDE_soln.mb.r_end**2* np.trapz(np.sin(theta_grid) * 0.5*np.real(q_phi[0,:]*np.conjugate(tau_phi_r[0,:])+q_theta[0,:]*np.conjugate(tau_theta_r[0,:])),x=theta_grid,axis=0)

surf_2 = 2*np.pi * ek* PDE_soln.mb.r_end**2* np.trapz(np.sin(theta_grid) * 0.5*np.real(q_phi[0,:]*tau_phi_r[0,:]+q_theta[0,:]*tau_theta_r[0,:]),x=theta_grid,axis=0)
surf_3 = 2*np.pi * ek* PDE_soln.mb.r_end**2* np.trapz(np.sin(theta_grid) * 0.5*np.imag(q_phi[0,:]*tau_phi_r[0,:]+q_theta[0,:]*tau_theta_r[0,:]),x=theta_grid,axis=0)



en_a = 1/4*(np.abs(q_r)**2+np.abs(q_phi)**2+np.abs(q_theta)**2)
en_b = 1/4 * (q_r**2+q_phi**2+q_theta**2)


kin_a = -2*np.pi*np.trapz(np.sin(theta_grid)*np.trapz(PDE_soln.mb.r_grid[:,np.newaxis]**2*en_a,x=PDE_soln.mb.r_grid,axis=0),x=theta_grid,axis=0)
kin_b =-2*np.pi*np.trapz(np.sin(theta_grid)*np.trapz(PDE_soln.mb.r_grid[:,np.newaxis]**2*en_b,x=PDE_soln.mb.r_grid,axis=0),x=theta_grid,axis=0)
print(kin_a,kin_b)
print()
print(surf_power,surf_2,surf_3)
print(tot_disp)
print(Ekman_disp(for_freq,1,ek,rad_ratio))
print(tot_en)
print()

mean_adv_r = 1/2 * np.real(q_r*np.conjugate(dr_q_r)+q_theta*np.conjugate(dtheta_q_r)+q_phi*np.conjugate(dphi_q_r))
mean_adv_theta = 1/2 * np.real(q_r*np.conjugate(dr_q_theta)+q_theta*np.conjugate(dtheta_q_theta)+q_phi*np.conjugate(dphi_q_theta))
mean_adv_phi = 1/2 * np.real(q_r*np.conjugate(dr_q_phi)+q_theta*np.conjugate(dtheta_q_phi)+q_phi*np.conjugate(dphi_q_phi))






fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=200)

if rad_ratio > 0.0:
    ss = np.array([[PDE_soln.mb.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[PDE_soln.mb.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])

if rad_ratio == 0.0:
    ss = np.array([[PDE_soln.mb.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[PDE_soln.mb.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])




freq = calc_freq(2,1,1)

print('freq',freq)
eig_r = calc_q_r(2,1,freq,PDE_soln.mb.r_grid*(1-rad_ratio),theta_grid)
eig_theta = calc_q_theta(2,1,freq,PDE_soln.mb.r_grid*(1-rad_ratio),theta_grid)
eig_phi = calc_q_phi(2,1,freq,PDE_soln.mb.r_grid*(1-rad_ratio),theta_grid)


norm = calc_norm(2,1,freq)/(1-rad_ratio)**3

inn_field = q_r * np.conjugate(eig_r) + q_theta * np.conjugate(eig_theta) +q_phi * np.conjugate(eig_phi)

inn = -2*np.pi/norm * np.trapz(np.sin(theta_grid)*np.trapz(inn_field*PDE_soln.mb.r_grid[:,np.newaxis]**2,x=PDE_soln.mb.r_grid,axis=0),x=theta_grid,axis=0)

print(np.abs(inn))

#field = np.real(q_phi*np.exp(1j*for_freq*(0.08758749*1e4-np.pi/2)))
#field = np.real(inn*eig_phi)
#field = np.real(q_phi*np.exp(1j*for_freq*(0*np.pi/2/for_freq+np.pi/2/for_freq)))

field = en

vmin = np.min(field)
vmax = np.max(field)



s_c = np.zeros(n_theta)
z_c = np.zeros(n_theta)

disp_vals = np.zeros(n_theta)
vel_vals = np.zeros(n_theta)
for i in range(n_theta):
    
    r_c = np.sin(np.pi/6)/np.sin(np.pi/6+theta_grid[i])
    

    disp_vals[i] = np.interp(r_c,PDE_soln.mb.r_grid[::-1],ek*disp[::-1,i])
    s_c[i] = r_c*np.sin(theta_grid[i])
    z_c[i] = r_c*np.cos(theta_grid[i])
    vel_vals[i] = np.interp(r_c,PDE_soln.mb.r_grid[::-1],np.sqrt(np.abs(q_r)**2+np.abs(q_theta)**2+np.abs(q_phi)**2)[::-1,i])
    
    
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/near_polar/ek_6_disp.txt',disp_vals)
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/near_polar/ek_6_vel.txt',vel_vals)
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/near_polar/ek_6_theta.txt',theta_grid)
#ax.plot(s_c,z_c,color='white')

ang = 10**0.7*ek**(1/5)
s_val = np.sin(ang)/np.tan(np.arcsin(for_freq/2))
print(ang)
#ax.plot(np.sin(ang),np.cos(ang),marker='s',color='k')

#ax.plot([0,np.sin(ang)],[np.cos(ang)-s_val,np.cos(ang)])


max_idc = 0

while theta_grid[max_idc] < ang:
    max_idc += 1
max_idc += -1


rad_res = 50

val_arr = np.zeros((max_idc,rad_res))

rads_arr = np.zeros_like(val_arr)

for i in range(max_idc):
    
    
    l = (np.cos(ang)-np.sin(ang)/np.tan(np.arcsin(for_freq/2))) * for_freq/2 / np.sin(np.arcsin(for_freq/2)-theta_grid[i])
    L = np.cos(ang)/np.cos(theta_grid[i])
    
    rad_grid = np.linspace(l,L,rad_res)
    
    rads_arr[i,:] = rad_grid
    val_arr[i,:] = np.interp(rad_grid,PDE_soln.mb.r_grid[::-1],disp[::-1,i])
    
    
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/rads_arr.txt',rads_arr)
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/theta_grid.txt',theta_grid)
#np.savetxt('C:/Users/ianma/Desktop/sphrflow/revision_files/vals_arr.txt',val_arr)
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
    
    
    off=-5.5
    top = 0
    norm = colors.LogNorm(vmin = 10**(off)*vmax,vmax=10**(top)*vmax)
    levels = np.logspace(max_dec+off,max_dec+top,100)


    cmap = 'turbo'
step = 1
#norm = colors.TwoSlopeNorm(vcenter=0,vmin=vmin,vmax=vmax)
p=ax.contourf(ss[::step,::step],zz[::step,::step],field[::step,::step],levels=levels,cmap=cmap,norm=norm)
cbar = fig.colorbar(p)
#cbar.set_ticks([vmin,0,vmax])
#cbar.ax.set_yticklabels([round(vmin,5),0,round(vmax,5)])

cbar.set_ticks([1e-6,1e-4,1e-2,1])
#cbar.set_ticks([-1,-0.1,0.1,1])
#cbar.ax.set_yticklabels([round(vmin,5),0,round(vmax,5)])


ax.set_aspect('equal')
ax.axis('off')



#ax.plot(1.9755282,0,ms=1,marker='o')

#fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)

#ax.plot(PDE_soln.mb.r_grid,np.imag(q_theta[:,len(theta_grid)//2]))

'''
fig,ax = plt.subplots(1,1,figsize=(8,5),dpi=200)

idx = np.argmin(np.abs(theta_grid-np.pi/2))

field = np.real(q_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Real$\{q\cdot \hat{\phi}\}$',color='k')

field = np.imag(q_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Imag$\{q\cdot \hat{\phi}\}$',color='k',ls='dashed')

field = np.real(inn*eig_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Real$\{A_k q_{k}\cdot \hat{\phi}\}$',color='blue')

field = np.imag(inn*eig_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Imag$\{A_k q_{k}\cdot \hat{\phi}\}$',color='blue',ls='dashed')


field = np.real(q_phi-inn*eig_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Real$\{(q-A_k q_{k})\cdot \hat{\phi}\}$',color='red')

field = np.imag(q_phi-inn*eig_phi)
trans = np.trapz(np.sin(theta_grid[np.newaxis,:])*field*np.sin(theta_grid[np.newaxis,:]),x=theta_grid,axis=1)/np.trapz(np.sin(theta_grid[np.newaxis,:])*np.sin(theta_grid[np.newaxis,:])**2,x=theta_grid,axis=1)
ax.plot(LN_case.r_grid,trans,label=r'Imag$\{(q-A_k q_{k})\cdot \hat{\phi}\}$',color='red',ls='dashed')

ax.set_xlim(LN_case.r_start,LN_case.r_end)
#ax.plot(theta_grid,field[0,:])
ax.legend(frameon=False,fontsize=8)
ax.set_xlabel('radius, $r$')
'''
#print(LN_case.r_grid[30])
current, peak = tracemalloc.get_traced_memory()

print(current,peak)

tracemalloc.clear_traces()
'''
with open("conv_testing/ek_8_{:d}.txt".format(l_max), "a") as myfile:
    myfile.write("{:d}    {:.18f}    {:.18f}    {:.18f}    {:d}    {:.5f}\n".format(N,surf_power,tot_disp,tot_en,peak,sol_time))
'''