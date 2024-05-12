import numpy as np
import matplotlib.pyplot as plt
import time 

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf
from tools import vector_products as vp

nu = 1

# defining mesh
H = 10  # +1
h = 1/H
x = np.arange(0, 1+h, h)
y = np.arange(0, 1+h, h)
X, Y = np.meshgrid(x, y)
num_vertices = (H+1)**2
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T

f_1 = lambda x: (2 * (np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])) + (np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))
f_2 = lambda x: (2 * (np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])) + (np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))

gdg             = lambda vert0, vert1, x, h: - nu * vp.grad_dot_grad(  bf.phi_2d(vert0, x, h), bf.phi_2d(vert1, x, h))
fmpdf_x_phiquad = lambda vert0, vert1, x, h:        vp.func_mul_pdfunc(bf.phi_2d(vert0, x, h), bf.phi_2d(vert1, x, h), axis=0)
fmpdf_y_phiquad = lambda vert0, vert1, x, h:        vp.func_mul_pdfunc(bf.phi_2d(vert0, x, h), bf.phi_2d(vert1, x, h), axis=1)

blocks = []
"""mat_funcs = [gdg            , 0              , fmpdf_x_phiquad,
             0              , gdg            , fmpdf_y_phiquad,
             fmpdf_x_phiquad, fmpdf_y_phiquad, 0               ]
"""
mat_funcs = [gdg, fmpdf_x_phiquad, fmpdf_y_phiquad]

F = np.zeros(3*((H+1)**2))
#force_funcs = [f_1, f_2, 0]

# main calculation loop
t0 = time.time()
steps = int(H*2)
for k, func in enumerate(mat_funcs):
    block = np.zeros((num_vertices, num_vertices))

    if func == 0:
        blocks.append(block)
    else:
        for i, vert0 in enumerate(vertices):
            
            percentage = round(100 * ((i)/(len(vertices)-1)), 1)
            print(f"Block num {k+1}/{len(mat_funcs)} ! {percentage}% ! time running {round((time.time() - t0)/60, 2)} minutes  " , end="\r")
            
            ele_size = 2*h
            x0 = np.linspace(vert0[0]-ele_size, vert0[0]+ele_size, steps)
            y0 = np.linspace(vert0[1]-ele_size, vert0[1]+ele_size, steps)
            X0, Y0 = np.meshgrid(x0, y0) 
            
            for j, vert1 in enumerate(vertices):
                integrand1 = func(vert0, vert1, (X0, Y0), h)
                block[i, j] = np.trapz(np.trapz(integrand1, y0, axis=0), x0, axis=0)
                
        blocks.append(block)
        
t1 = time.time()
print(f"time taken: {round((t1-t0)/60, 2)} minutes                                                        ")

#reshaped_blocks = [blocks[(i*3):(i*3)+3] for i in range(3)]
zero = np.zeros((num_vertices, num_vertices))
reshaped_blocks = [[blocks[0], zero     , blocks[1]],
                   [zero     , blocks[0], blocks[2]],
                   [blocks[1], blocks[2], zero     ]]

A = np.block(reshaped_blocks)

for i, vert0 in enumerate(vertices):
    ele_size = 2*h
    x0 = np.linspace(vert0[0]-ele_size, vert0[0]+ele_size, steps)
    y0 = np.linspace(vert0[1]-ele_size, vert0[1]+ele_size, steps)
    X0, Y0 = np.meshgrid(x0, y0)

    integrand1           = bf.phi_2d(vert0, (X0, Y0), h) * f_1((X0, Y0))
    F[i]                 = np.trapz(np.trapz(integrand1, y0, axis=0), x0, axis=0)
    
    integrand2           = bf.phi_2d(vert0, (X0, Y0), h) * f_2((X0, Y0))
    F[int(((H+1)**2)/3) + i] = np.trapz(np.trapz(integrand2, y0, axis=0), x0, axis=0)


solution = np.linalg.lstsq(A, F, rcond=None)[0]

plt.matshow(A)
plt.colorbar()
u_1 = solution[           0 :     (H+1)**2].reshape((H+1), (H+1))
u_2 = solution[    (H+1)**2 : 2*((H+1)**2)].reshape((H+1), (H+1))
p   = solution[2*((H+1)**2) :             ].reshape((H+1), (H+1))

fig, ax = plt.subplots()
ax.quiver(X, Y, u_1, u_2)
fig, ax = plt.subplots()
ax.streamplot(X, Y, u_1, u_2)

plt.matshow(u_1)
plt.title("u_1")
plt.colorbar()
plt.matshow(u_2)
plt.title("u_2")
plt.colorbar()
plt.matshow(p)
plt.title("p")
plt.colorbar()
plt.show()