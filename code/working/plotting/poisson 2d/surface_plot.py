# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf
 
H = 10
h = 1/H
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T
X, Y = np.meshgrid(x, y)

solution = torch.load("poisson_2d_sol.pt")
u = solution.reshape(10, 10)
print(u.shape)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, u, cmap="bone")
plt.show()
