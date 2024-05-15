"""
Plotting of 2dim hat function
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf 

N = 31
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, bf.hat_2d((0.5, 0.5), (X, Y), .25), cmap="viridis")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'$\phi_{\frac{1}{2},\frac{1}{2}}(x,y)$')
#plt.savefig("hat_basis_2d.png", dpi=500)
plt.show()