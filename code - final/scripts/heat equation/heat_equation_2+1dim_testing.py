"""
This a file is for testing whether our script for the heat equation works with an chosen forcig function

heat equation with forcing function

--------- in LateX code ---------------

$f(x,y,t) = (2\pi^2t + 1)sin(\pi x)sin(\pi y)$

and solution

$u(x, y, t) = t sin(\pi x)sin(\pi y)$

--------------------------------------

"""

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
from tools import norms as nrm

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
u_0 = [0 for x in vertices] # inital condtions at time=0
u_sols = [u_0] # list for keeping track of evolution of system
A_sols = []

def force(x, t):
    return ((2*(np.pi**2)*t) + 1) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

# functions for calculation the entries of matrices
def f(vert, x, u, time):
    return (u + (dt * force(x, time))) * bf.phi_2d(vert, x, h)

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
            F[j] = F[j] + np.trapz(np.trapz( f(vert0, (X0, Y0), u, t*dt) , y0, axis=0), x0, axis=0)
            
            for vert1 in ele:
                i = vertices.index(vert1)
                A[j, i] = A[j, i] + np.trapz(np.trapz( a(vert0, vert1, (X0, Y0)) , y0, axis=0), x0, axis=0)
                
                
    solution_t = np.matmul(np.linalg.pinv(A), F)
    u_sols.append(solution_t)
    A_sols.append(A)
    
    
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")

# creating mesh with higher resolution for ploting
N = 4*H
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

func_u_sols = []
for i in range(len(u_sols)):
    func_u_sols.append(bf.conv_sol(u_sols[i], (X, Y), bf.phi_2d, vertices, h))

"""--------------- PLOTTING --------------------"""

# plot and animation
len_u_sols = len(func_u_sols)
frn = len_u_sols
fps = frn//3

def change_plot(frame_number, func_u_sols, plot):
   plot[0].remove()  
   plot[0] = ax.plot_surface(X, Y, func_u_sols[frame_number], cmap="plasma", vmin=0, vmax=.5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot = [ax.plot_surface(X, Y, func_u_sols[0])]
ax.set_zlim(0, .5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y, t)")
ax.set_title(r"$ \frac{\partial u}{\partial t} - \nabla^2 u = (2 \pi ^2 t + 1)sin(\pi x)sin(\pi y)$ with $u(x,y,0) = 0$")

ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(func_u_sols, plot), interval=1000 / fps)
#ani.save('animation.gif',writer='PillowWriter',fps=fps, dpi=400)
plt.show()

# plot of matrix for first iteration
plt.matshow(A_sols[0])
plt.title("Matrix for first iteration")
plt.colorbar()
plt.show()

# creating a sequence of images
plt.clf()
iter1 = 0
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.axes.set_zlim3d(bottom=0, top=.5) 
ax.plot_surface(X, Y, func_u_sols[iter1], cmap="plasma", vmin=0, vmax=.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(f'$u(x, y, {iter1*dt})$')
#plt.savefig(f"heat_equ_iter{iter1}", dpi=400)

plt.clf()
iter2 = int(len_u_sols*(1/3))
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.axes.set_zlim3d(bottom=0, top=.5) 
ax.plot_surface(X, Y, func_u_sols[iter2], cmap="plasma", vmin=0, vmax=.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(f'$u(x, y, {iter2*dt})$')
#plt.savefig(f"heat_equ_iter{iter2}", dpi=400)

plt.clf()
iter3 = int(len_u_sols*(2/3))
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.axes.set_zlim3d(bottom=0, top=.5) 
ax.plot_surface(X, Y, func_u_sols[iter3], cmap="plasma", vmin=0, vmax=.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(f'$u(x, y, {iter3*dt})$')
#plt.savefig(f"heat_equ_iter{iter3}", dpi=400)

plt.clf()
iter4 = time_steps
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.axes.set_zlim3d(bottom=0, top=.5) 
ax.plot_surface(X, Y, func_u_sols[iter4], cmap="plasma", vmin=0, vmax=.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(f'$u(x, y, {iter4*dt})$')
#plt.savefig(f"heat_equ_iter{iter4}", dpi=400)

#error check at t=0 
math_sol = lambda x, t: t*np.sin(np.pi *x[0])*np.sin(np.pi *x[1])
error = nrm.l_squ_norm_2d(func_u_sols[0] - math_sol((x, y), 0))
print(f"error = {error}")