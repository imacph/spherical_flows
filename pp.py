# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:34:18 2024

@author: ianma
"""
import numpy as np
import matplotlib.pyplot as plt




fig,ax=plt.subplots(1,1,figsize=(8,5),dpi=200)

ek = 1e-4
freq = np.loadtxt('freqs_ek_4_shell.txt')
disp = np.loadtxt('disp_ek_4_shell.txt')
kin = np.loadtxt('kin_ek_4_shell.txt')

tt = np.linspace(-1,1,300)
th = np.trapz(np.sqrt(np.abs(freq[:,np.newaxis]+2*tt[np.newaxis,:]))*(-tt[np.newaxis,:]**4+2*tt[np.newaxis,:]**3-2*tt[np.newaxis,:]+1),x=tt,axis=1)


ax.plot(freq,disp/ek**(1/2),label='$E=10^{-4}$; $\eta = 0.5$',lw=0.7)
#ax1.plot(freq,kin/0.5**2)


ek = 1e-4
freq = np.loadtxt('freqs_ek_4_shell_1.txt')
disp = np.loadtxt('disp_ek_4_shell_1.txt')

tt = np.linspace(-1,1,300)
th = np.trapz(np.sqrt(np.abs(freq[:,np.newaxis]+2*tt[np.newaxis,:]))*(-tt[np.newaxis,:]**4+2*tt[np.newaxis,:]**3-2*tt[np.newaxis,:]+1),x=tt,axis=1)


ax.plot(freq,disp/ek**(1/2),label='$E=10^{-4}$; $\eta = 0.0$',lw=0.7)
'''
ek = 1e-4
freq = np.loadtxt('freqs_ek_4_shell_3.txt')
disp = np.loadtxt('disp_ek_4_shell_3.txt')

tt = np.linspace(-1,1,300)
th = np.trapz(np.sqrt(np.abs(freq[:,np.newaxis]+2*tt[np.newaxis,:]))*(-tt[np.newaxis,:]**4+2*tt[np.newaxis,:]**3-2*tt[np.newaxis,:]+1),x=tt,axis=1)


ax.plot(freq,disp/ek**(1/2),label='$E=10^{-4}$; $\eta = 0.3$',lw=0.7)

ek = 1e-4
freq = np.loadtxt('freqs_ek_4_topo.txt')
disp = np.loadtxt('disp_ek_4_topo.txt')
kin = np.loadtxt('kin_ek_4_topo.txt')

tt = np.linspace(-1,1,300)
th = np.trapz(np.sqrt(np.abs(freq[:,np.newaxis]+2*tt[np.newaxis,:]))*(-tt[np.newaxis,:]**4+2*tt[np.newaxis,:]**3-2*tt[np.newaxis,:]+1),x=tt,axis=1)


#ax.plot(freq,disp/ek**(1/2))
ax.plot(freq,th*np.pi/np.sqrt(2),label='B.L. theory',lw=0.7)
#ax1.plot(freq,kin)

ek = 1e-5
freq = np.loadtxt('freqs_ek_5_topo.txt')
disp = np.loadtxt('disp_ek_5_topo.txt')


'''

th = np.trapz(np.sqrt(np.abs(freq[:,np.newaxis]+2*tt[np.newaxis,:]))*(-tt[np.newaxis,:]**4+2*tt[np.newaxis,:]**3-2*tt[np.newaxis,:]+1),x=tt,axis=1)

ax.plot(freq,disp/ek**(1/2),label='$E=10^{-5}$; $\eta = 0.0$',lw=0.7)

#ax.set_yscale('log')

ax.set_xlim(-2,2)
#ax1.set_xlim(-2,2)
ax.legend(frameon=False,loc=2)
ax.tick_params(axis='both', which='major', labelsize=8)
#ax1.tick_params(axis='both', which='major', labelsize=8)
#ax1.set_yscale('log')
ax.set_xlabel('libration frequency, $\lambda$')
ax.set_ylabel('Dissipation/$E^{1/2}$')

#fig.savefig('spec_eta_5.png',bbox_inches='tight')


p =2
freq0 = 0.184
freq0 += -0.698*np.sqrt(1e-5)
freq0 += -0.1936*(1e-6)**(1/3)


#$freq0 = -1.445+0.626*np.sqrt(1e-5)
#freq0 = -1.4406+0.809*np.sqrt(1e-8)
print(freq0)

ek = 1e-4
freq = np.loadtxt('freqs_ek_4_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_4_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_4_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*10,color='blue')

ek = 1e-5
freq = np.loadtxt('freqs_ek_5_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_5_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_5_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*10,color='red')

ek = 1e-6
freq = np.loadtxt('freqs_ek_6_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_6_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_6_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*10,color='k')


p =3


freq0 = -1.445+0.626*np.sqrt(1e-5)
freq0 = -1.4406+0.809*np.sqrt(1e-8)

ek = 1e-4
freq = np.loadtxt('freqs_ek_4_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_4_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_4_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='blue')

ek = 1e-5
freq = np.loadtxt('freqs_ek_5_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_5_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_5_eta_5_peak_{:d}.txt'.format(p))

ax.plot(freq,disp*2*np.pi,color='red')

ek = 1e-6
freq = np.loadtxt('freqs_ek_6_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_6_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_6_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='k')


ek = 1e-7
freq = np.loadtxt('freqs_ek_7_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_7_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_7_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='purple')

ek = 1e-8
freq = np.loadtxt('freqs_ek_8_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_8_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_8_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='goldenrod')
ax.set_yscale('log')


freq0= 1.315626
freq0 = 1.315626 + 0.896*np.sqrt(1e-7)

freq0 += 2.633*np.sqrt(1e-8)

p =1



ek = 1e-4
freq = np.loadtxt('freqs_ek_4_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_4_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_4_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='blue')

ek = 1e-5
freq = np.loadtxt('freqs_ek_5_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_5_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_5_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='red')

ek = 1e-6
freq = np.loadtxt('freqs_ek_6_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_6_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_6_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='k')

ek = 1e-7
freq = np.loadtxt('freqs_ek_7_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_7_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_7_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='purple')

ek = 1e-8
freq = np.loadtxt('freqs_ek_8_eta_5_peak_{:d}.txt'.format(p))
disp = np.loadtxt('disp_ek_8_eta_5_peak_{:d}.txt'.format(p))
kin = np.loadtxt('kin_ek_8_eta_5_peak_{:d}.txt'.format(p))


ax.plot(freq,disp*2*np.pi,color='goldenrod')
#ax.set_yscale('log')
