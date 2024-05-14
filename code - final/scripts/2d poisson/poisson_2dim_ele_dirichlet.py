"""
Script solving the 2d poisson problem with dirichlet boudary conditions. 

With manfactured forcing function 

--------- in LateX code-----------

$f(x, y) = -2 \pi^2 sin(\pi x)sin(\pi y)$

and solution

$u(x, y) = -sin(\pi x)sin(\pi y)

"""

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
from tools import norms as nrm

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
    return -2*(np.pi**2)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

def f(vert, x):
    return bf.phi_2d(vert, x, h) * force(x)

def a(vert0, vert1, x):
    return vp.grad_dot_grad_phi2d(vert0, vert1, x, h)


t0 = time.time()
for k, ele in enumerate(elements):
    x0 = np.linspace(ele[0][0], ele[3][0], steps)
    y0 = np.linspace(ele[0][1], ele[3][1], steps)
    X0, Y0 = np.meshgrid(x0, y0)
    
    percentage = round(100 * ((k)/(len(elements)-1)), 1)
    print(f"{percentage}%", end="\r")
    
    for vert0 in ele:
        j = vertices.index(vert0)
        F[j] = F[j] + np.trapz(np.trapz( f(vert0, (X0, Y0)) , y0, axis=0), x0, axis=0)
        
        for vert1 in ele:
            i = vertices.index(vert1)
            A[j, i] = A[j, i] + np.trapz(np.trapz( a(vert0, vert1, (X0, Y0)) , y0, axis=0), x0, axis=0)
        
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")

# this is soling the matrix equation Ax = F
solution = np.matmul(np.linalg.pinv(A), F)

# the rest is plotting
N = 3*H
x0 = np.linspace(0, 1, N)
y0 = np.linspace(0, 1, N)
X0, Y0 = np.meshgrid(x0, y0) 

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X0, Y0, bf.conv_sol(solution, (X0, Y0), bf.phi_2d, vertices, h), cmap="viridis")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title(r"$ \nabla^2 u(x,y) = -2 \pi^2sin(\pi x)sin(\pi y) $")
#plt.savefig(f"(Poisson_2d)_(vertex_num_{H**2})", dpi=500)

plt.matshow(A)
plt.colorbar()
#plt.savefig(f"(Poisson_2d)_(mat_A)_(vertex_num_{H**2})", dpi=500)
plt.show()

#error check
math_sol = lambda x: -np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
ele_sol = bf.conv_sol(solution, (X0, Y0), bf.phi_2d, vertices, h)
print(f"L squared norm error = {nrm.l_squ_norm_2d(ele_sol - math_sol((X0, Y0)), (x0, y0))}")