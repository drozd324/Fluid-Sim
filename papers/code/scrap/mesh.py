import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

H = 4
h = 1/H
a = np.arange(0, 1, 1/H)
b = np.arange(0, 1, 1/H)
num_vertices = H**2
vertices = np.array(np.meshgrid(a, b)).reshape(2, -1).T

points = vertices
tri = Delaunay(points)

print(points)
print(vertices)

plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()