"""
Plotting of 2dim quadratic basis functions
"""


import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf 



N = 200
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

h = 0.1

pair_1 = (0.5, 0.6)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, bf.quad_2d(pair_1, (X, Y), h), cmap="viridis")
#plt.savefig(f"quad_basis_2d_{pair_1}.png")
plt.show()

pair_2 = (0.5, 0.5)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, bf.quad_2d(pair_2, (X, Y), h), cmap="viridis")
#plt.savefig(f"quad_basis_2d_{pair_2}.png")
plt.show()

pair_3 = (0.4, 0.6)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, bf.quad_2d(pair_3, (X, Y), h), cmap="viridis")
#plt.savefig(f"quad_basis_2d_{pair_3}.png")
plt.show()