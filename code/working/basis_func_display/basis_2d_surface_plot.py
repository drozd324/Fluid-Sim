import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf 

H = 3 + (2*4)
a = np.linspace(0, 1, H)
b = np.linspace(0, 1, H)
h = a[1] - a[0]
vertices = np.array(np.meshgrid(a, b)).reshape(2, -1).T

N = 200
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

X, Y = np.meshgrid(x, y)

u = 0
for l, i in enumerate(vertices):
    u = u + bf.psi_2d(i, [X, Y], h)
    
#u = bf.psi_2d([], [X, Y], h)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, u, cmap="viridis")
plt.show()