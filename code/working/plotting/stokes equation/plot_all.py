import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

H = 51 # +1
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
h = x[1] - x[0]
X, Y = np.meshgrid(x, y)
num_vertices = (H+1)**2
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T

data = torch.load("stokes.pt")
A = data["A"]
F = data["F"]
u_1 = data["u_1"]
u_2 = data["u_2"]
p = data["p"]

def conv(solution, x, y, basis_func):
    ans = 0
    for i, ele in enumerate(vertices):
        ans = ans + (solution[i] * basis_func(ele, (x, y), h))
    return ans

N = 100
x0 = np.linspace(0, 1, N)
y0 = np.linspace(0, 1, N)
n = x0[1] -  x0[0]
X0, Y0 = np.meshgrid(x0, y0)

u_1_resh = conv(u_1.reshape(((H+1)**2)), X0, Y0, bf.phi_2d)
u_2_resh = conv(u_2.reshape(((H+1)**2)), X0, Y0, bf.phi_2d)
p_resh = conv(p.reshape(((H+1)**2)), X0, Y0, bf.phi_2d)

fig, ax = plt.subplots()
ax.quiver(X0, Y0, u_1_resh, u_2_resh)

fig, ax = plt.subplots()
ax.streamplot(X0, Y0, u_1_resh, u_2_resh)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_title("u_1")
ax.plot_surface(X0, Y0, u_1_resh, cmap="viridis")

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_title("u_2")
ax.plot_surface(X0, Y0, u_2_resh, cmap="viridis")

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_title("p")
ax.plot_surface(X0, Y0, p_resh, cmap="viridis")

plt.show()