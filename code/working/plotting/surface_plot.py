# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

 
H = 11
h = 1/H
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T
X, Y = np.meshgrid(x, y)

#u = np.sin(X * np.pi) * np.sin(np.pi * Y)
#u = np.exp(-(X-(1/2))**2 - (Y-(1/2))**2) - np.exp(-1/4)
#u = (Y**2) * np.sin(np.pi*x)
u = -Y*X*(X-1)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, u, cmap="viridis")
plt.show()
