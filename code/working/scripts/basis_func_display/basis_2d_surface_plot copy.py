import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf 

N = 99
n = 1/N
x = np.arange(0, 1+n, n)
print(len(x))
y = np.arange(0, 1+n, n)
X, Y = np.meshgrid(x, y)

H = 2*2
h = 1/H
a = np.arange(0, 1+h, h)
b = np.arange(0, 1+h, h)
vertices = np.array(np.meshgrid(a, b)).reshape(2, -1).T

#u = np.sin(np.pi*X) * np.cos(np.pi*Y)
#u = np.sin(np.pi*X) * np.sin(np.pi*Y)

#u = X*(X-1)*Y*(Y-1)
#u = -(2*X - 1)*(Y**2)*((y/3) - (1/2))

#exp = lambda x: np.exp(-(x-(1/2))**2) - np.exp(-1/4)

#plt.plot(x, exp(x))
#plt.show()

#exp_2d = lambda x, y: exp(x) * exp(y)

#u = exp_2d(X, Y)

#u_1 = (X**2)*((1-X)**2)*Y*(1-Y)*(1-(2*Y))
#u_2 = -(X)*(1-X)*(1-(2*Y))*(Y**2)*((1-Y)**2)

#p   = lambda x: x[0]*((2*x[0])-1) * x[1]*((2*x[1])-1)
p   = lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])


fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, p((X, Y)), cmap="viridis")
plt.show()
