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

nu = 1
steps = 20 # accuracy or steps in integrals

# defining mesh
H = 3 + (2*3)
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y)
h = x[1] - x[0]
vertices = (np.array(np.meshgrid(x, y)).reshape(2, -1).T).tolist()

lin_elements = [[[x[i]  , y[j] ],
                [x[i+1], y[j]  ], 
                [x[i]  , y[j+1]],
                [x[i+1], y[j+1]]] for i in range(H-1) for j in range(H-1)]

quad_elements = []
for i in list(np.arange(0, H-1, 2)):
    for j in list(np.arange(0, H-1, 2)):
        
        element = []
        for a in range(0, 3):
            for b in range(0, 3):
                element.append([x[i+a], y[j+b]])
                
        quad_elements.append(element)

u_1 = lambda x: (x[0]**2)*((1-x[0])**2)*x[1]*(1-x[1])*(1-(2*x[1]))
u_2 = lambda x: -(x[0])*(1-x[0])*(1-(2*x[1]))*(x[1]**2)*((1-x[1])**2)

dx_u1 = lambda x: 2*x[0]*(1-x[0])*(1 - (2*x[0]))*x[1]*(1-x[1])*(1 - (2*x[1]))
dy_u2 = lambda x: - dx_u1(x)

ddx_u1 = lambda x: 2*y[1]*(1-x[1])*(1-(2*x[1]))*( ((1-x[0])*1-(2*x[0])) - (x[0]*(1-(2*x[0])) - (2*x[0]*(1-x[0])) ))
ddy_u1 = lambda x: (x[0]**2) * ((1 - x[0])**2) * ( - (2*(1 - (2*x[1]))) - (4*(1 - x[1])) + (4*x[1]) )

ddx_u2 = lambda x: (x[1]**2) * ((1 - x[1])**2) * ( (2*(1 - (2*x[0]))) + (4*(1 - x[0])) - (4*x[0]) )
ddy_u2 = lambda x: - 2*y[0]*(1-x[0])*(1-(2*x[0]))*( ((1-x[1])*1-(2*x[1])) - (x[1]*(1-(2*x[1])) - (2*x[1]*(1-x[1])) ))

dx_p = lambda x: 2*np.pi*np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
dy_p = lambda x: 2*np.pi*np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])

# declaring appropirate forcing functions
f_1 = lambda x: 0
f_2 = lambda x: -1

# declaring appropriate funtions for block matrices
gdg        = lambda vert0, vert1, x, h: nu * vp.grad_dot_grad(bf.psi_2d(vert0, x, h), bf.psi_2d(vert1, x, h) , h)
hat_dx_psi = lambda vert0, vert1, x, h:  bf.phi_2d(vert0, x, h) * (np.gradient(bf.psi_2d(vert1, x, h), h)[1])
hat_dy_psi = lambda vert0, vert1, x, h:  bf.phi_2d(vert0, x, h) * (np.gradient(bf.psi_2d(vert1, x, h), h)[0])

t0 = time.time()
# block matrix calculations
mat_blocks = []
mat_funcs  = [gdg, hat_dx_psi, hat_dy_psi]
for ka, a in enumerate(mat_funcs):
    BLOCK = np.zeros((H**2, H**2))

    for k, ele in enumerate(quad_elements):
        x0 = np.linspace(ele[0][0], ele[-1][0], steps)
        y0 = np.linspace(ele[0][1], ele[-1][1], steps)
        X0, Y0 = np.meshgrid(x0, y0)
        
        percentage = round(100 * ((k)/(len(quad_elements)-1)), 1)
        print(f"mat_{ka} {percentage}%", end="\r")
        
        for vert0 in ele:
            j = vertices.index(vert0)
            for vert1 in ele:
                i = vertices.index(vert1)
                BLOCK[j, i] = BLOCK[j, i] + np.trapz(np.trapz( a(vert0, vert1, (X0, Y0), h) , y0, axis=0), x0, axis=0)
    
    mat_blocks.append(BLOCK)
    
# block vector calculations
vect_blocks = []
force_funcs = [f_1, f_2]
for kf, f in enumerate(force_funcs):
    BLOCK = np.zeros((H**2))

    for k, ele in enumerate(quad_elements):
        x0 = np.linspace(ele[0][0], ele[-1][0], steps)
        y0 = np.linspace(ele[0][1], ele[-1][1], steps)
        X0, Y0 = np.meshgrid(x0, y0)
        
        percentage = round(100 * ((k)/(len(quad_elements)-1)), 1)
        print(f"force_{kf} {percentage}%", end="\r")
        
        for vert0 in ele:
            j = vertices.index(vert0)
            BLOCK[j] = BLOCK[j] + np.trapz(np.trapz( (bf.hat_2d(vert0, (X0, Y0), h)) * f((X0, Y0)) , y0, axis=0), x0, axis=0)
    
    vect_blocks.append(BLOCK)

t1 = time.time()
print(f"time taken: {round((t1-t0)/60, 2)} minutes                                                        ")

# assembly of blocks
zero_mat = np.zeros((H**2, H**2))
zero_vect = np.zeros((H**2))

reshaped_mat_blocks = [[mat_blocks[0], zero_mat     , -mat_blocks[1]],
                       [zero_mat     , mat_blocks[0], -mat_blocks[2]],
                       [mat_blocks[1], mat_blocks[2], zero_mat     ]]

reshaped_vect_blocks = [vect_blocks[0], vect_blocks[1], zero_vect]

A = np.block(reshaped_mat_blocks)
F = np.block(reshaped_vect_blocks)

solution = np.matmul(np.linalg.pinv(A), F)


# creating mesh with higher resolution for ploting
N = 40
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# dissasembling solution vector
u_1 = bf.conv_sol(solution[       0 : H**2    ], (X, Y), bf.psi_2d, vertices, h)
u_2 = bf.conv_sol(solution[H**2     : 2*(H**2)], (X, Y), bf.psi_2d, vertices, h)
p   = bf.conv_sol(solution[2*(H**2) :         ], (X, Y), bf.hat_2d, vertices, h)

# saving data just in case
import torch
data = {"A"  : A,
        "F"  : F,
        "u_1": u_1,
        "u_2": u_2,
        "p"  : p   }
torch.save(data, f"stokes{H}.pt")


plt.matshow(A)
plt.colorbar()
plt.title("Matrix")
plt.savefig("Vector_pde_matrix", dpi=500)

fig, ax = plt.subplots()
ax.quiver(X, Y, u_1, u_2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$\vec{u}(x, y)$")
plt.savefig("vector fiel", dpi=500)

fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$\vec{u}(x, y)$")
ax.streamplot(X, Y, u_1, u_2)
plt.savefig("stream", dpi=500)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$u_x(x, y)$")
ax.set_title("Horizontal Velocity")
ax.plot_surface(X, Y, u_1, cmap="viridis")
plt.savefig("horizontal velocity", dpi=500)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$u_y(x, y)$")
ax.set_title("Vertical Velocity")
ax.plot_surface(X, Y, u_2, cmap="viridis")
plt.savefig("vertical velocity", dpi=500)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("P(x, y)")
ax.set_title("Pressure")
ax.plot_surface(X, Y, p, cmap="viridis")
plt.savefig("pressure", dpi=500)


plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
new_h = x[1] - x[0]
ax.plot_surface(X, Y, np.gradient(u_1, new_h)[1] + np.gradient(u_2, new_h)[0], cmap="viridis")
plt.show()