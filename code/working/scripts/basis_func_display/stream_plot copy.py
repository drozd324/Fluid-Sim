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


#grad_x, grad_y = np.gradient(f(X, Y), h, h)
gamma = lambda x: (1 - np.cos(2*np.pi*x[0])) * (1 - np.cos(2*np.pi*x[1]))

#u_1 = lambda x: np.gradient(gamma(x), h)[1]
#u_2 = lambda x: np.gradient(gamma(x), h)[0]

u_1 = lambda x: 2*np.pi*(1 - np.cos(2*np.pi*x[0]))*np.sin(2*np.pi*x[1])
u_2 = lambda x: -2*np.pi*(1 - np.cos(2*np.pi*x[1]))*np.sin(2*np.pi*x[0])

dx_u1 = lambda x: np.gradient(u_2(x), h)[1]
dy_u2 = lambda x: np.gradient(u_1(x), h)[0]

fig, ax = plt.subplots()
ax.quiver(X, Y, u_1((X, Y)), u_2((X, Y)))
fig, ax = plt.subplots()
ax.streamplot(X, Y, u_1((X, Y)), u_2((X, Y)))
plt.show()
