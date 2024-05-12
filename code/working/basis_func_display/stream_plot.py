"""
General test script for plotting vector fields and 2d functions
"""

import numpy as np
import matplotlib.pyplot as plt

H = 10
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
h = x[1] - x[0]
vertices = (np.array(np.meshgrid(x, y)).reshape(2, -1).T).tolist()
elements = [[[x[i]  , y[j]  ],
             [x[i+1], y[j]  ], 
             [x[i]  , y[j+1]], 
             [x[i+1], y[j+1]]] for i in range(H-1) for j in range(H-1)]
X, Y = np.meshgrid(x, y)

def f(x, y):
    return x*(x-1)*y*(y-1)

u_1 = lambda x: (x[0]**2)*((1-x[0])**2)*x[1]*(1-x[1])*(1-(2*x[1]))
u_2 = lambda x: -(x[0])*(1-x[0])*(1-(2*x[1]))*(x[1]**2)*((1-x[1])**2)

#grad_x, grad_y = np.gradient(f(X, Y), h, h)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, u_1((X, Y)), cmap="viridis")
plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, u_2((X, Y)), cmap="viridis")
plt.show()

fig, ax = plt.subplots()
ax.quiver(X, Y, u_1((X, Y)), u_2((X, Y)))
fig, ax = plt.subplots()
ax.streamplot(X, Y, u_1((X, Y)), u_2((X, Y)))
plt.show()
