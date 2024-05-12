import numpy as np
import matplotlib.pyplot as plt
import time 

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf
from tools import vector_products as vp

steps = 200 # accuracy or steps in integrals
 
# defining mesh
H = 7
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
h = x[1] - x[0]
vertices = (np.array(np.meshgrid(x, y)).reshape(2, -1).T).tolist()
elements = [[[x[i]  , y[j]  ],
             [x[i+1], y[j]  ], 
             [x[i]  , y[j+1]], 
             [x[i+1], y[j+1]]] for i in range(H-1) for j in range(H-1)]

 
A = np.zeros((H**2, H**2))
F = np.zeros((H**2))

def force(x):
    x, y = x
    return 2*y

def n(x):
    x, y = x
    return  x*(x-1)
    
def f(vert, x):
    return bf.phi_and_hat_2d(vert, x, h, [0]) * force(x)

def a(vert0, vert1, x):
    return vp.gdg_phi_and_hat_2d(vert0, vert1, x, h, [0])

def neumann(vert, x):
    x, y = x
    int1 = np.trapz( n((x, 0)) * bf.phi_and_hat_2d(vert, (x, 0), h, [0]) , x, axis=0)
    int2 = np.trapz( n((x, 1)) * bf.phi_and_hat_2d(vert, (x, 1), h, [0]) , x, axis=0)
    return int1 + int2

t0 = time.time()
for k, ele in enumerate(elements):
    x0 = np.linspace(ele[0][0], ele[3][0], steps)
    y0 = np.linspace(ele[0][1], ele[3][1], steps)
    X0, Y0 = np.meshgrid(x0, y0)
    
    percentage = round(100 * ((k)/(len(elements)-1)), 1)
    print(f"{percentage}%", end="\r")
    
    for vert0 in ele:
        j = vertices.index(vert0)
        F[j] = F[j] + np.trapz(np.trapz( f(vert0, (X0, Y0)) , y0, axis=0), x0, axis=0) - neumann(vert0, (x0, y0))
        
        for vert1 in ele:
            i = vertices.index(vert1)
            A[j, i] = A[j, i] + np.trapz(np.trapz( a(vert0, vert1, (X0, Y0)) , y0, axis=0), x0, axis=0)
        
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")

# this is soling the matrix equation Ax = F
solution = np.matmul(np.linalg.pinv(A), F)

# the rest is plotting
N = H
x0 = np.linspace(0, 1, N)
y0 = np.linspace(0, 1, N)
X0, Y0 = np.meshgrid(x0, y0)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X0, Y0, bf.conv_sol(solution, (X0, Y0), bf.hat_2d, vertices, h), cmap="viridis")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title(r"$ \nabla^2u(x,y) = 2y $ with neumann bdry $n(x,y) = x(x-1)$")
plt.savefig(f"(Poisson_2d)_(neumann)_(vertex_num_{H**2})")

plt.matshow(A)
plt.colorbar()
plt.savefig(f"(Poisson_2d)_(mat)_(neumann)_(vertex_num_{H**2})")
plt.show()