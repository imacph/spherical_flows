# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
fig,ax=plt.subplots(1,1,figsize=(8,5),dpi=200)

kin_spec = np.loadtxt('kin_spec_4_topo.txt')

n_freq,n_l = np.shape(kin_spec)

freqs = np.linspace(-2,2,n_freq)


ll = np.linspace(2,n_l+1,n_l)
print(ll)


freq_grid,ll_grid = np.meshgrid(freqs,ll,indexing='ij')


field = kin_spec


vmin = np.min(field)
vmax = np.max(field)
norm = colors.LogNorm(vmin = vmin,vmax= vmax)
levels = np.logspace(np.log10(vmin),np.log10(vmax),100)

ax.contourf(freq_grid,ll_grid,field,norm = norm,levels=levels,cmap='inferno')

#ax.set_yscale('log')
ax.set_ylim(2,n_l+1)
#fig,ax=plt.subplots(1,1,figsize=(8,5),dpi=200)

freqs = [-0.23192505,  1.23192505,-0.10179042,  0.88424897, -1.09256839,
         -0.10179042,  0.88424897, -1.09256839,  1.64344318,-0.05770957,  
         0.68967539, -0.80583384,  1.3358916,  -1.45580853, -0.05770957,  
         0.68967539, -0.80583384,  1.3358916,  -1.45580853,  1.79378495]

for i in range(len(freqs)):
    ax.axvline(freqs[i],color='white',alpha=0.5)