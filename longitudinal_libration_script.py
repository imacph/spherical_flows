import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from Matrix_builder import Matrix_builder
from Boundary_rhs_builder import Boundary_rhs_builder
from PDE_matrix_frame import PDE_matrix_frame,Spatial_representation



 
'resolution and symmetry parameters'
N =40 # number of radial grid points in soln.
l_max =40# maximum spherical harmonic degree in soln.
rad_ratio = 0. # spherical shell aspect ratio
m = 0 # azimuthal symmetry order

'libration parameters'
ek = 10**-4 # ekman number
Re = 1 # reynolds number
for_freq = 1. # forcing frequency

# calculates the appropriate libration amplitude
eps = Re*np.sqrt(ek)*(1-rad_ratio)

# information about BCs to pass to solver
bc_list = [['tor','t',1,eps*2*np.sqrt(np.pi/3)/(1-rad_ratio)**2]]
'matrix construction and inverse problem solution'
t0 = time() 

# building the PDE matrix
matrix_builder = Matrix_builder(N,rad_ratio,m,l_max,radial_method='finite_difference')
PDE_mat = PDE_matrix_frame(matrix_builder.gen_PDE_matrix('tor',for_freq,ek),matrix_builder,ek,for_freq)

# building the libration forcing RHS
rhs_builder = Boundary_rhs_builder(N,rad_ratio,m,l_max)
rhs_builder.gen_rhs(bc_list,'tor')

# solving inverse problem and formatting solution arrays
PDE_soln = PDE_mat.solve_sys(rhs_builder.rhs)
PDE_soln.process_soln('tor')

# the time to complete the solution
sol_time = time()-t0
print('time to solve: {:.2f}s'.format(sol_time))

# calculating colatitude angle theta grid for spatial representation of solution
n_theta = 4*PDE_soln.mb.n_l
theta_min = 0
theta_max = np.pi
theta_grid = 0.5 * (np.cos(np.linspace(1,n_theta,n_theta)*np.pi/(n_theta+1))[::-1] * (theta_max-theta_min) + (theta_max+theta_min))

#theta_grid = np.loadtxt('theta.txt')
theta_grid[0] += 1e-6
theta_grid[-1] += -1e-6
n_theta = len(theta_grid)

spat_rep = Spatial_representation(theta_grid,PDE_soln)

# calculating velocity field and its gradients
PDE_soln.calc_vel_field(spat_rep)
PDE_soln.calc_vel_grad(spat_rep)

'shortening names'
q_r,q_theta,q_phi = spat_rep.q_r,spat_rep.q_theta,spat_rep.q_phi

dr_q_r,dtheta_q_r,dphi_q_r = spat_rep.dr_q_r,spat_rep.dtheta_q_r,spat_rep.dphi_q_r 
dr_q_theta,dtheta_q_theta,dphi_q_theta = spat_rep.dr_q_theta,spat_rep.dtheta_q_theta,spat_rep.dphi_q_theta
dr_q_phi,dtheta_q_phi,dphi_q_phi = spat_rep.dr_q_phi,spat_rep.dtheta_q_phi,spat_rep.dphi_q_phi 


spat_rep.calc_bulk_dissipation()
spat_rep.calc_total_kin()
spat_rep.calc_surface_power(full_calc=True)
spat_rep.calc_advection()


print(spat_rep.total_dissipation)
print(spat_rep.power_cmb,spat_rep.power_cmb_real_coeff,spat_rep.power_cmb_imag_coeff)


field = np.real(q_phi*np.exp(1j*for_freq*(0)))

#field = np.sqrt(np.abs(q_r)**2+np.abs(q_theta)**2+np.abs(q_phi)**2)

#field = spat_rep.dissipation
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=400)

if rad_ratio > 0.0:
    ss = np.array([[PDE_soln.mb.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[PDE_soln.mb.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])

if rad_ratio == 0.0:
    ss = np.array([[PDE_soln.mb.r_grid[i]*np.sin(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])
    zz = np.array([[PDE_soln.mb.r_grid[i]*np.cos(theta_grid[j]) for j in range(n_theta)] for i in range(N+1)])




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




if vmin < 0:
    norm = colors.SymLogNorm(vmin = vmin1, vmax = vmax1, linthresh=thres)
    levels = -np.logspace(np.log10(thres),min_dec,100)[::-1]
    levels = np.append(levels,np.linspace(-thres,thres,100)[1:])
    levels = np.append(levels,np.logspace(np.log10(thres),max_dec,100)[1:])
    cmap = 'seismic'
elif vmin >= 0:
    
    
    off=-9
    top = -0
    norm = colors.LogNorm(vmin = 10**(off)*vmax,vmax=10**(top)*vmax)
    levels = np.logspace(max_dec+off,max_dec+top,100)


    cmap = 'hot'
step = 1
#norm = colors.TwoSlopeNorm(vcenter=0,vmin=vmin,vmax=vmax)
p=ax.contourf(ss[::step,::step],zz[::step,::step],field[::step,::step],levels=levels,cmap=cmap,norm=norm)
#cbar = fig.colorbar(p)
#cbar.set_ticks([vmin,0,vmax])
#cbar.ax.set_yticklabels([round(vmin,5),0,round(vmax,5)])

#cbar.set_ticks([1e-6,1e-4,1e-2,1])
#cbar.set_ticks([-1,-0.1,0.1,1])
#cbar.ax.set_yticklabels([round(vmin,5),0,round(vmax,5)])


ax.set_aspect('equal')
ax.axis('off')


plt.show()