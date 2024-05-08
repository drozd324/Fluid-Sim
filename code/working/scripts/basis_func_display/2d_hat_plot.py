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

fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, bf.hat_2d((0.5, 0.5), (X, Y), .5), cmap="viridis")
plt.savefig("hat_basis_2d.png")