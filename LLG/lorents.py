import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D



init   = [1.0,0.0,0.0]
t_span = [0.0,50.0]
t_eval = np.linspace(*t_span,3000) # time for sampling
p=10
r=2
b=8./3.
def lorenz(t,X,p,r,b):
    x,y,z = X
    return [-p*x+p*y, -x*z+r*x-y, x*y-b*z]
sol = solve_ivp(lorenz,t_span,init,method='RK45',t_eval=t_eval,args=(p,r,b,))

fig = plt.figure(); ax = Axes3D(fig)
ax.plot(sol.y[0,:],sol.y[1,:],sol.y[2,:],'k-',lw=0.5)
