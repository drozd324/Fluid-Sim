import numpy as np
import matplotlib.pyplot as plt

H = 100
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

grad_x, grad_y = np.gradient(f(X, Y), h)
#grad_x, grad_y = np.gradient(f(X, Y), h, h)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, f(X, Y), cmap="viridis")
plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, grad_x, cmap="viridis")
plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, grad_y, cmap="viridis")
plt.show()

plt.plot(x, f(x, .5))
plt.plot(x, np.gradient(f(x, .5), h))
plt.show()