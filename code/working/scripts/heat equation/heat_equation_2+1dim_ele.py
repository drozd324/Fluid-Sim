import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf
from tools import vector_products as vp

steps = 100 # accuracy or steps in integrals
 
# defining mesh
H = 7
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y)
h = x[1] - x[0]
vertices = (np.array(np.meshgrid(x, y)).reshape(2, -1).T).tolist()
elements = [[[x[i]  , y[j]  ],
             [x[i+1], y[j]  ], 
             [x[i]  , y[j+1]], 
             [x[i+1], y[j+1]]] for i in range(H-1) for j in range(H-1)]
 
# declaring time stepping nececities
dt = .01 # change in time
time_steps = 50 # amount of iterations we want
u_0 = [np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]) for x in vertices] # inital condtions at time=0
u_sols = [u_0] # list for keeping track of evolution of system

def force(x):
    return 0

# functions to calculate entries of matrices
def f(vert, x, u):
    return (u + (dt * force(x))) * bf.phi_2d(vert, x, h)

def a(vert0, vert1, x):
    return (bf.phi_2d(vert0, x, h)*bf.phi_2d(vert1, x, h)) + (dt*(vp.grad_dot_grad_phi2d(vert0, vert1, x, h)))

t0 = time.time()
for t in range(time_steps): #time stepping loop
    A = np.zeros((H**2, H**2))
    F = np.zeros((H**2))
    
    # main code, iterating over elements and calculating entries of matrix
    for k, ele in enumerate(elements):
        # creating domain of element to interate over
        x0 = np.linspace(ele[0][0], ele[3][0], steps)
        y0 = np.linspace(ele[0][1], ele[3][1], steps)
        X0, Y0 = np.meshgrid(x0, y0)
        
        u = bf.conv_sol(u_sols[t], (X0, Y0), bf.phi_2d, vertices, h)
        
        percentage = round(100 * ((k)/(len(elements)-1)), 1)
        print(f"time {t} and {percentage}%", end="\r")
        
        for vert0 in ele:
            j = vertices.index(vert0)
            F[j] = F[j] + np.trapz(np.trapz( f(vert0, (X0, Y0), u) , y0, axis=0), x0, axis=0)
            
            for vert1 in ele:
                i = vertices.index(vert1)
                A[j, i] = A[j, i] + np.trapz(np.trapz( a(vert0, vert1, (X0, Y0)) , y0, axis=0), x0, axis=0)
                
                
    solution_t = np.matmul(np.linalg.pinv(A), F)
    u_sols.append(solution_t)
    
    
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")

func_u_sols = []
for i in range(len(u_sols)):
    func_u_sols.append(np.array(u_sols[i]).reshape(H, H))

#import torch
#torch.save(func_u_sols, "heat_equation_ele.pt")

# plot and animation
len_u_sols = len(func_u_sols)
frn = len_u_sols
fps = frn//3

def change_plot(frame_number, func_u_sols, plot):
   plot[0].remove()  
   plot[0] = ax.plot_surface(X, Y, func_u_sols[frame_number], cmap="plasma", vmin=0, vmax=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot = [ax.plot_surface(X, Y, func_u_sols[0])]
ax.set_zlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y, t)")
ax.set_title(r"$ \frac{\partial u}{\partial t} = \nabla^2 u $ with $u(x,y,0) = sin(\pi x)sin(\pi y)$")

#ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(func_u_sols, plot), interval=1000 / fps)
plt.show()