# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

H = 10
h = 1/H
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
num_vertices = H**2
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T
X, Y = np.meshgrid(x, y)

#solution = torch.load("poisson_2d_sol.pt")
solution = torch.load("temp_conv_func.pt")

def u(x, list):
    ans = 0
    for i, vert in enumerate(vertices):
        ans = ans + (list[i] * bf.phi_2d(vert, x, h))
    return ans

import torch
linear_sol = torch.load("poisson_2d_sol.pt")
u_linear = linear_sol.reshape(H, H)
a = np.linspace(0, 1, 100)
b = np.linspace(0, 1, 100)
u_interp = scipy.interpolate.interp2d(x, y, u_linear, kind="linear")
A, B = np.meshgrid(a, b)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(A, B, u_interp(a, b), cmap="bone")
#ax.plot_surface(X, Y, u((X, Y), solution), cmap="bone")
#ax.plot_surface(X, Y, u_linear, cmap="bone")
plt.show()
