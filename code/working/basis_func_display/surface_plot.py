"""
General test script for plotting of 2d functions
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf 

N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
h = x[1] - x[0]
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection ='3d')

u_1 = lambda x: (x[0]**2)*((1-x[0])**2)*x[1]*(1-x[1])*(1-(2*x[1]))
u_2 = lambda x: -(x[0])*(1-x[0])*(1-(2*x[0]))*(x[1] **2)*((1-x[1])**2)

#dx_u1 = lambda x: 2*x[0]*(1-x[0])*(1 - (2*x[0]))*x[1]*(1-x[1])*(1 - (2*x[1]))
#dy_u2 = lambda x: - dx_u1(x)
dx_u1 = lambda x: np.gradient(u_1(x), h)[1]
dy_u2 = lambda x: np.gradient(u_2(x), h)[0]

tot = lambda x: dx_u1(x) + dy_u2(x)

ddx_u1 = lambda x: 2*y[1]*(1-x[1])*(1-(2*x[1]))*( ((1-x[0])*1-(2*x[0])) - (x[0]*(1-(2*x[0])) - (2*x[0]*(1-x[0])) ))
ddy_u1 = lambda x: (x[0]**2) * ((1 - x[0])**2) * ( (-2*(1 - (2*x[1]))) - (4*(1 - x[1])) + (4*x[1]) )

ddx_u2 = lambda x: (x[1]**2) * ((1 - x[1])**2) * ( (2*(1 - (2*x[0]))) + (4*(1 - x[0])) + (4*x[0]) )
ddy_u2 = lambda x: - 2*y[0]*(1-x[0])*(1-(2*x[0]))*( ((1-x[1])*1-(2*x[1])) - (x[1]*(1-(2*x[1])) - (2*x[1]*(1-x[1])) ))

p = lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

dx_p = lambda x: 2*np.pi*np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
#dy_p = lambda x: 2*np.pi*np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])

#dy_u1, dx_u1 = np.gradient(u_1((X, Y)), h)

#dx_u1 = lambda x: np.gradient(u_1(x), h, axis=1)
#dy_u2 = lambda x: np.gradient(u_2(x), h, axis=0)

#dx_p, dy_p = np.gradient(p((X, Y)), h)

"""
gamma = lambda x: (1 - np.cos(2*np.pi*x[0])) * (1 - np.cos(2*np.pi*x[1]))

u_1 = lambda x: np.gradient(gamma(x), h, axis=0)
u_2 = lambda x: np.gradient(gamma(x), h, axis=1)

dx_u1 = lambda x: np.gradient(u_1(x), h, axis=0)
dy_u2 = lambda x: np.gradient(u_2(x), h, axis=1)

tot = lambda x: dx_u1(x) + dy_u2(x)
"""

#ax.plot_surface(X, Y, u_1((X, Y)), cmap="viridis")

#ax.plot_surface(X, Y, dx_u1((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, dx_u1((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, dx_u1, cmap="viridis")

#ax.plot_surface(X, Y, p((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, dx_p((X, Y)), cmap="viridis")
ax.plot_surface(X, Y, dx_u1((X, Y)) + dy_u2((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, dy_p, cmap="viridis")

#plt.savefig("hat_basis_2d.png")
#ax.plot_surface(X, Y, -Y*X*(X-1), cmap="viridis")

plt.show()