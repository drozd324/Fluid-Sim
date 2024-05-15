import numpy as np
import matplotlib.pyplot as plt



N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

mu = 1 # viscocity
u_1 = lambda x: 2*np.pi*(1 - np.cos(2*np.pi*x[0]))*np.sin(2*np.pi*x[1])
u_2 = lambda x: -2*np.pi*(1 - np.cos(2*np.pi*x[1]))*np.sin(2*np.pi*x[0])
p = lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot_surface(X, Y, u_1((X, Y)), cmap="viridis")
#plt.savefig("pressure", dpi=500)
plt.show()
