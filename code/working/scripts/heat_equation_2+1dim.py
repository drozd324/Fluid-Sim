import numpy as np
import matplotlib.pyplot as plt
import time

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

# defining mesh
H = 8
h = 1/H
box_len = 3
a = np.linspace(-box_len, box_len, H)
b = np.linspace(-box_len, box_len, H)
num_vertices = H**2
vertices = np.array(np.meshgrid(a, b)).reshape(2, -1).T

# time step nonsense
dt = .001
time_steps = 10
u_sols = []
u_0 = np.array([np.exp(-(x[0]**2 + x[1]**2)) for x in list(vertices)])
u_sols.append(u_0)

# defining the solution u
def u(t, x):
    ans = 0
    for i, ele in enumerate(vertices):
        ans = ans + (u_sols[t][i] * bf.phi_2d(ele, x, h))
    return ans

def f(x):
    """forcing function

    Args:
        x (tuple): variable

    Returns:
        float:
    """
    return 0

A = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

# preparing functions for later
def func1(i, x, t):
    return (u(t, x) + (dt * f(x))) * bf.phi_2d(i, x, h)

def func2(i, a, x):
    summand1 = bf.phi_2d(i, x, h) * bf.phi_2d(a, x, h)
    summand2_1 = bf.grad_phi(i[0], x[0], h) * bf.phi(i[1], x[1], h)      * bf.grad_phi(a[0], x[0], h) * bf.phi(a[1], x[1], h)
    summand2_2 = bf.phi(i[0], x[0], h)      * bf.grad_phi(i[1], x[1], h) * bf.phi(a[0], x[0], h)      * bf.grad_phi(a[1], x[1], h)
    return summand1 + (dt*(summand2_1 + summand2_2))

t0 = time.time()
"""cmax = np.amax(u_0)
plt.imshow(u_0.reshape(H, H))
plt.clim(0, cmax)
plt.colorbar()
plt.pause(.01)
"""
for t in range(time_steps): #time stepping loop
    # main calculation loop 
    for i, ele0 in enumerate(vertices):
        
        # defining integration limits
        step = 100
        x0 = np.linspace(ele0[0]-h, ele0[0]+h, step)
        y0 = np.linspace(ele0[1]-h, ele0[1]+h, step)
        X0, Y0 = np.meshgrid(x0, y0) 
        
        integrand0 = func1(ele0, (X0, Y0), t)
        F[i] = np.trapz(np.trapz(integrand0, y0, axis=0), x0, axis=0)
        
        percentage = round(100 * ((i)/(num_vertices-1)), 1)
        print(f"{percentage}% non linear haha", end="\r")

        for j, ele1 in enumerate(vertices[0:i+1]):
            
            integrand1 = func2(ele0, ele1, (X0, Y0))
            A[i, j] = np.trapz(np.trapz(integrand1, y0, axis=0), x0, axis=0)
            A[j, i] = A[i, j]
    
    solution_t = np.linalg.lstsq(A, F, rcond=None)[0]
    u_sols.append(solution_t)
    
    """plt.clf()
    plt.imshow(solution_t.reshape(H, H))
    plt.clim(0, cmax)
    plt.colorbar()
    plt.pause(.01)"""
    
    #print(u_sols[t] - u_sols[t+1])

t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")
plt.show()

func_u_sols = []
for i in range(len(u_sols)):
    func_u_sols.append(u_sols[i].reshape(H, H))
import torch
torch.save(func_u_sols, "heat_equation.pt")