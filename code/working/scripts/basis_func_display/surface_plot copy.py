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

#gamma = lambda x: (1 - np.cos(2*np.pi*x[0])) * (1 - np.cos(2*np.pi*x[1]))

#u_1 = lambda x: np.gradient(gamma(x), h)[1]
#u_2 = lambda x: np.gradient(gamma(x), h)[0]


u_1 = lambda x: 2*np.pi*(1 - np.cos(2*np.pi*x[0]))*np.sin(2*np.pi*x[1])
u_2 = lambda x: -2*np.pi*(1 - np.cos(2*np.pi*x[1]))*np.sin(2*np.pi*x[0])

dx_u1 = lambda x: np.gradient(u_1(x), h)[1]
dy_u2 = lambda x: np.gradient(u_2(x), h)[0]

ax.plot_surface(X, Y, u_1((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, u_2((X, Y)), cmap="viridis")
#ax.plot_surface(X, Y, dx_u1((X, Y)) + dy_u2((X, Y)), cmap="viridis")

#plt.savefig("hat_basis_2d.png")
#ax.plot_surface(X, Y, -Y*X*(X-1), cmap="viridis")

plt.show()