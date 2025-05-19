import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

fig,ax = plt.subplots(1,1,figsize=(6,5),dpi=200)

N = 200

pts = cheb.chebpts1(N)
mapped_pts = 1/2*(pts+1)
nn = np.linspace(0,N-1,N)


eval_mat = cheb.chebval(pts,np.eye(N+4)).T
df1_mat = 2*cheb.chebval(pts,cheb.chebder(np.eye(N+4),m=1)).T
df2_mat = 4*cheb.chebval(pts,cheb.chebder(np.eye(N+4),m=2)).T
df3_mat = 8*cheb.chebval(pts,cheb.chebder(np.eye(N+4),m=3)).T
df4_mat = 16*cheb.chebval(pts,cheb.chebder(np.eye(N+4),m=4)).T

B = -2 * (nn+2)/(nn+3)
D = -(1+B)

eval_mat_f = eval_mat[:,:-4] - eval_mat[:,2:-2]
df1_mat_f = df1_mat[:,:-4] - df1_mat[:,2:-2]
df2_mat_f = df2_mat[:,:-4] - df2_mat[:,2:-2]
df3_mat_f = df3_mat[:,:-4] - df3_mat[:,2:-2]
df4_mat_f = df4_mat[:,:-4] - df4_mat[:,2:-2]


eval_mat_g = eval_mat[:,:-4]  + B[np.newaxis,:]* eval_mat[:,2:-2] + D[np.newaxis,:]* eval_mat[:,4:]
df1_mat_g = df1_mat[:,:-4]  + B[np.newaxis,:]* df1_mat[:,2:-2] + D[np.newaxis,:]* df1_mat[:,4:]
df2_mat_g = df2_mat[:,:-4]  + B[np.newaxis,:]* df2_mat[:,2:-2] + D[np.newaxis,:]* df2_mat[:,4:]
df3_mat_g = df3_mat[:,:-4]  + B[np.newaxis,:]* df3_mat[:,2:-2] + D[np.newaxis,:]* df3_mat[:,4:]
df4_mat_g = df4_mat[:,:-4]  + B[np.newaxis,:]* df4_mat[:,2:-2] + D[np.newaxis,:]* df4_mat[:,4:]




ax.plot(mapped_pts,df2_mat_f[:,0])