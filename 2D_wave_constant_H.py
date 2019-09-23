import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation

plt.close('all')


def c1fun(K,k1,k2,dk,g,H,f):
    
    r = K.shape[1] # rank
    c= np.zeros((r,r,2))
    
    for i in range(r):
        for j in range(r):
            c[i,j,0] = np.sum(g*H*k1*K[:,i]*K[:,j]/omega)*(dk**2)
            c[i,j,1] = np.sum(g*H*k2*K[:,i]*K[:,j]/omega)*(dk**2)
    return c
            

def d2fun(X, dx, D):
    
    r = X.shape[1] # rank
    Nx = int(np.sqrt(X.shape[0])) # number of discretised points
    
    d = np.empty((r,r,2), dtype=complex, order='F') # empty array for d2
    
    # move to fourier space
    Xf = np.fft.fft2(X.reshape(Nx,Nx,rx, order='F'),axes=(0,1)).reshape(Nx**2,rx, order='F')
    
    for i in range(r):
        for j in range(r):
            d[i,j,0] = np.sum(np.conj(Xf[:,i])*(Xf[:,j])*1j*xf1*(dx**2)/(Nx**2))
            d[i,j,1] = np.sum(np.conj(Xf[:,i])*(Xf[:,j])*1j*xf2*(dx**2)/(Nx**2))
    
    return np.real(d)


g = 1 #gravitational acceleration
H = 1 # Height of water
f = 1e-4 #spin of earth


# Grid for space and velocity
Nx = 64
D = 2*np.pi
Nk = 32 
max_k = 5
min_k = -5
k0 = 3
x, dx = np.linspace(0,D, Nx, retstep = True)
k, dk = np.linspace(min_k, max_k, Nk, retstep =True)

x1, x2  = np.meshgrid(x, x)
k1, k2 = np.meshgrid(k, k)

k1 = k1.reshape(Nk**2,order='F')
k2 = k2.reshape(Nk**2,order='F')

# fix parameters
sig = 0.3
delt = 0.1


# Initial conditions
# Reshape with order F because of the way the meshgrid is constructed. 
# Alternatively we could transpose the meshgrids and use order C which is default.
X = []
X.append(((1/(2*np.pi*sig**2))*(np.exp(-((x1-D/2)**2 + (x2-D/2)**2)/(2*sig**2))).reshape(Nx**2, order='F')))
#X.append(((1/(2*np.pi*sig**2))*(np.exp(-((x1-D/2)**2 + (x2-3*D/4)**2)/(2*sig**2))).reshape(Nx**2, order='F')))
#X.append(((1/(2*np.pi*sig**2))*(np.exp(-((x1-D/2)**2 + (x2-D/4)**2)/(2*sig**2))).reshape(Nx**2, order='F')))
#X.append(np.random.normal(size=(Nx**2)))
#X.append(0.1*(np.sin(x1+x2)).reshape(Nx**2,order='F'))
#X.append(0.1*(np.sin(x1+x2)).reshape(Nx**2,order='F'))

X0 = np.asarray(X).T
X0, R0 = np.linalg.qr(X0)
X0 = X0/dx
R0 = R0*dx
X = X0
#X = np.squeeze(np.asarray(X))

K = []
#K = (np.exp(-(k1**2 +k2**2)/(2*delt**2))).reshape(Nk**2, order='F')
#K.append(((1/(2*np.pi*delt**2))*np.exp(-((k1+k0)**2 + (k2+1)**2)/(2*delt**2))).reshape(Nk**2, order='F'))
K.append(((1/(2*np.pi*delt**2))*np.exp(-((k1-k0)**2 + (k2-1)**2)/(2*delt**2))).reshape(Nk**2, order='F'))

#K.append(np.random.normal(size=(Nk**2)))
#K.append((delt*np.sin(k1)+delt*np.sin(3*k2)).reshape(Nk**2,order='F'))


K0 = np.asarray(K).T
K0, R1 = np.linalg.qr(K0)
K0 = K0/dk
R1 = R1*dk
K = K0


rk = K.shape[1]
rx = X.shape[1]

rank_control = np.identity(rk)
#rank_control[1,1]=0
#rank_control[2,2]=0
#S = [[4,0],[0,1]]
S0= R0 @ rank_control @ R1.T#np.asarray([[1,0],[0,1]]) @R1.T
S= S0



a = X @ S @ K.T # initial a
#X, S, KT = np.linalg.svd(a)
#K = KT.T
#
#r=3
#rx = r
#rk = rx
#
## Do regularisation
#S = np.diag(S[0:r])
#X = X[:,0:r]
#K = K[:,0:r]
#a = X @ S @ K.T

a_k_int = (np.sum(a, axis=1)*dk**2).reshape(Nx,Nx,order='F')
a_x_int = (np.sum(a, axis=0)*dx**2).reshape(Nk,Nk,order='F')

vmink = 2*a_k_int.min()
vmaxk = 2*a_k_int.max()
vminx = 2*a_x_int.min()
vmaxx = 2*a_x_int.max()

#plt.figure(1)
fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(111, projection='3d')
#ax1 = Axes3D(fig1)
#ax1 = fig1.gca(projection='3d')
ims = []
im = plt.imshow(np.flipud(a_x_int), animated=True)#, vmin = vminx,  vmax = vmaxx)
#im = ax1.plot_surface(k1.reshape(Nk,Nk,order='F'),k2.reshape(Nk,Nk,order='F'),np.flipud(a_x_int),animated=True,color='b')
ims.append([im])
#plt.colorbar()

#fig2 = plt.figure()
fig2 = plt.figure(2)
#ax2 = fig2.add_subplot(111, projection='3d')
#ax2 = fig2.gca(projection='3d')
ims2 = []
im2 = plt.imshow(np.flipud(a_k_int), animated=True)#, vmin = vmink,  vmax = vmaxk)
#im2 = ax2.plot_surface(x1,x2,np.flipud(a_k_int),animated=True,color='b')
ims2.append([im2])
#plt.colorbar()

omega = np.sqrt(f**2 + g*H*(k1**2+k2**2))

xf = np.fft.fftfreq(Nx, D/(2*np.pi*Nx))
xf1, xf2 = np.meshgrid(xf,xf)
xf1 = xf1.reshape(Nx**2, order='F')
xf2 = xf2.reshape(Nx**2, order='F')
    


Nsteps = 50
dt = dx


for timestep in range(Nsteps):
    
    M0 = X @ S
    M0hat = np.fft.fft2(M0.reshape(Nx,Nx,rk, order='F'),axes=(0,1)).reshape(Nx**2,rk, order='F')
    
    c1 = c1fun(K,k1,k2,dk, g, H, f)
    
    C1 = np.empty((rk,rk,Nx**2), dtype=complex)
    
    for i in range(rk):
        for j in range(rk):
            C1[i,j,:] = c1[i,j,0]*1j*xf1 + c1[i,j,1]*1j*xf2
            
    
    
    Mhat_step = np.empty((Nx**2,rk), dtype=complex)
    for i in range(Nx**2):
        Mhat_step[i,:] = la.expm(-dt*C1[:,:,i]) @ (M0hat[i,:].T)
    
    M_step = np.real(np.fft.ifft2(Mhat_step.reshape(Nx,Nx,rk, order='F'),axes=(0,1)).reshape(Nx**2,rk, order='F'))
    X1, Shat1 = np.linalg.qr(M_step)
    X1 = X1/dx
    Shat1 = Shat1*dx
    
    d2 = d2fun(X1, dx, D)
    S_factor = np.tensordot(d2,c1,axes=[2,2])
    
    Stild0 = np.empty((rx,rk))
    
    for i in range(rx):
        for j in range(rk):
            Stild0[i,j] = Shat1[i,j] + dt*np.sum(Shat1*S_factor[i,:,j,:]) # or is it +dt
          
    D2 = np.empty((rk,rk,Nk**2))
    
    
    for i in range(rk):
        for j in range(rk):
            D2[i,j,:] = ((d2[i,j,0]*k1 + d2[i,j,1]*k2)*g*H/omega).reshape(Nk**2)
    
    L0 = Stild0 @ K.T
    
    Lstep = np.empty((Nk**2, rk))
    for i in range(Nk**2):
        Lstep[i,:] = la.expm(-dt*D2[:,:,i].T) @ (L0[:,i])
        
    K1, S1T = np.linalg.qr(Lstep)
    K1 = K1/dk
    S1T = S1T*dk
    
    S1 = S1T.T
    astep = X1 @ S1 @ K1.T
    
    X = X1
    S = S1
    K = K1
    
    # rank in k-direction
    rk = K1.shape[1]
    # rank in x-direction
    rx = X1.shape[1]
    
    a_k_int = (np.sum(astep, axis=1)*dk**2).reshape(Nx,Nx,order='F')
    a_x_int = (np.sum(astep, axis=0)*dx**2).reshape(Nk,Nk,order='F')
    
    plt.figure(1)
    im = plt.imshow(np.flipud(a_x_int), animated=True)
    ims.append([im])
    plt.figure(2)
    im2 = plt.imshow(np.flipud(a_k_int), animated=True)
    ims2.append([im2])



# animation of imshows
animation.time = 2000 # length of animation in milliseconds
interval = animation.time/(Nsteps+1)
ani = animation.ArtistAnimation(plt.figure(1), ims, interval=3*interval, blit=True,
                                repeat_delay=0)

# ani.save('dynamic_images.mp4')

animation.time = 2000 # length of animation in milliseconds
interval = animation.time/(Nsteps+1)
ani2 = animation.ArtistAnimation(plt.figure(2), ims2, interval=3*interval, blit=True,
                                repeat_delay=0)

# ani.save('dynamic_images.mp4')
