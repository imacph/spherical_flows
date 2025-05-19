import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors






fig,ax=plt.subplots(1,1,figsize=(7,5),dpi=200)


R =7000


nk = 300
nt = 300
kk = np.logspace(-3,1,nk)
tt = np.linspace(0,2*np.pi,nt)

vals = np.zeros((nk,nt))

for i in range(nt):
    
    for j in range(nk):
        
        k = kk[j]
        t = tt[i]
        
        vals[j,i] = -2*k**2
        if k/2/np.pi*R*(np.cos(t)-np.exp(-2*np.pi/k/np.sqrt(2))*np.cos(t - 2*np.pi/k/np.sqrt(2))) + 2 < 0:
            
            
            vals[j,i] += np.sqrt(-(k/2/np.pi*R*(np.cos(t)-np.exp(-2*np.pi/k/np.sqrt(2))*np.cos(t - 2*np.pi/k/np.sqrt(2))) + 2))
            
            
k_grid,t_grid = np.meshgrid(kk,tt,indexing='ij')


vmin = np.min(vals)
vmax = np.max(vals)

if vmax > 0:
    norm = colors.TwoSlopeNorm(0,vmin=np.min(vals),vmax=np.max(vals))

    n = 200
    levs = np.linspace(vmin,0+vmin/n,n)
    print(levs)
    levs = np.concatenate((levs,np.linspace(0,vmax,10)))
    print(levs)
    
    
    p=ax.contourf(t_grid,k_grid,vals,norm=norm,cmap='seismic',levels=levs)
    
    #ax.set_yscale('log')
    ax.contour(t_grid,k_grid,vals,levels=[0])


else: 
    p=ax.contourf(t_grid,k_grid,vals,cmap='seismic',levels=300,norm=colors.SymLogNorm(vmin=vmin,vmax=vmax,linthresh=0.001))

ax.set_ylim(0,10)

fig.colorbar(p,label='Modal growth rate')
ax.set_xlabel('Rotational/librational time (t)')
ax.set_ylabel('Modal wave-number (k)')