import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Sets the fontsize for all figures.
plt.rcParams.update({'font.size': 18})

def LTUpdate(U0, S0, Vh0, delta_A):
    # This function performs one step of the dynamical 
    # low-rank approximation updating algorithm.
    
    V0 = np.transpose(Vh0)
    K1 = (U0@S0) + (delta_A@V0)
    U1, S1hat = np.linalg.qr(K1)
    S0tild = S1hat - (np.transpose(U1)@delta_A@V0)
    L1 = (V0@np.transpose(S0tild)) + (np.transpose(delta_A)@U1)
    V1, S1h = np.linalg.qr(L1)
    return U1, S1h.T, V1.T

def gaussian_true(sigma, t, x, y, x0, y0):
    # This function creates a gaussian centred at (x0,y0) with variance sigma.
    
    A = (1/(2*np.pi*(2*t+sigma**2)))*np.exp(-(1/(2*(2*t+sigma**2)))*((x-x0)**2 + (y-y0)**2))
    
    return A


# Rank of the approximation
r = 1

# Sets the value of pi to pi so we don't have to use numpy everytime.
pi  = np.pi

# X and Y meshgrid

D = 1 # Size of space
M = 128 # Number of gridspaces

x1, dx = np.linspace(0, D, M, retstep=True)
x2 = np.linspace(0, D, M)
X1, X2 = np.meshgrid(x1,x2)


# Initial Condition
sig = 0.5 # Variance of the initial condition.
A0 = (1/(2*pi*sig**2))*np.exp(-(1/(2*sig**2))*((X1[1:M-1,1:M-1]-0.5)**2 + (X2[1:M-1,1:M-1]-0.5)**2))

# Creates empty list to fill with true solutions.
A = []
A.append(A0)

# Discretisation matrix
disc_mat = np.diag(-2*np.ones(M-2)) + np.diag(np.ones(M-3),k=-1) + np.diag(np.ones(M-3),k=1)
disc_mat = disc_mat*(1/dx**2)

# Final Time
T = 0.05

# Number used to set the size of the time-step, such that C>h/dx^2.
C = 0.25
h = C*(dx**2) # Size of the time-step.

# Vector of time-steps, which runs up as close to T as 
# possible using multiples of h.
t = np.arange(0,T,h) 

N = len(t) # Number of  time-steps.

# Initial U0, S0, Vh0
U0, S0, V0h = np.linalg.svd(A[0])



#  Initialise list of matrices U,S,Vh
S = []
U = []
Vh = []

# Truncate the SVD to rank r
Sreg = np.diag(S0[0:r])
Ureg = U0[0:,0:r]
Vreg = V0h[0:r,:]

# Initial low-rank SVD
S.append(Sreg)
U.append(Ureg)
Vh.append(Vreg)

# Initial Y0, the low-rank approximation.
Y = []
Y.append(U[0]@ S[0]@ Vh[0])

# Error
# Create empty list for errors.
error = []
error.append(np.linalg.norm(A[0]-Y[0])/np.linalg.norm(A[0]))

# Sets the maximum and minimum values of the initial condition.
# This will be used in plotting.
vmin = Y[0].min()
vmax = Y[0].max()


# Loop over time
for i in range(N-1):
    
    # Dynamical low-rank approximation
    
    # Calculates delta A for use  in LTUpdate.
    delta_A = h*((disc_mat @ Y[i]) + (Y[i] @ np.transpose(disc_mat)))
    
    # calls the LTUpdate function, to update U,S and V.
    U1, S1, V1h = LTUpdate(U[i], S[i], Vh[i], delta_A)
    
    # Add new matrices to their respective lists.
    U.append(U1)
    S.append(S1)
    Vh.append(V1h)
    Y.append(U1 @ S1 @ V1h)
    
    # True solution at the current time-step.
    A.append(gaussian_true(sig,t[i+1],X1[1:M-1,1:M-1],X2[1:M-1,1:M-1],0.5,0.5))
    
    # Error between numerical and exact
    error.append(np.linalg.norm(A[i+1]-Y[i+1])/np.linalg.norm(A[i+1]))


# Plots
    
printno = [0, 500, 1500, N-1] # List of time-steps to plot for.

# The loop plots and saves the relevant figures.
for p in printno:
    plt.figure()
    plt.imshow(Y[p], vmin = vmin,  vmax = vmax, extent=[0,D,0,D])
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.savefig('heat_rank%.0f_%.0f.png'%(r,p),bbox_inches='tight')
    
# Plots the error against time.
plt.figure()
plt.plot(t, error)
plt.xlabel('Time')
plt.ylabel('Error Norm $|A(t)-Y(t)|/|A(t)|$')