import numpy as np
import time 
import scipy
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

# defining mesh
H = 10
h = 1/H
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
num_vertices = H**2
vertices = np.array(np.meshgrid(x, y)).reshape(2, -1).T
X, Y = np.meshgrid(x, y)

def g(x):
    return -1

J = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

"""def f(u, phi, x):
    integrand0 = np.multiply((u + 1), np.multiply(np.gradient(u), np.gradient(phi(x)))) - np.multiply(g(x), phi(x))
    return np.trapz(np.trapz(integrand0, x, axis=0), y, axis=0)
"""

def f_integrand(u, vert, x):
    grad_phi_x = np.gradient(bf.phi_2d(vert, x, h), axis=0)
    grad_phi_y = np.gradient(bf.phi_2d(vert, x, h), axis=1)
    grad_u_x = np.gradient(u, axis=0)
    grad_u_y = np.gradient(u, axis=1)    
    dot = np.multiply(grad_u_x, grad_phi_x) + np.multiply(grad_u_y, grad_phi_y)

    residual = np.multiply(g(x), bf.phi_2d(vert, x, h))
    
    non_lin_part = np.square(grad_u_x) + np.square(grad_u_y) 
    
    return -(np.multiply(non_lin_part, dot) - residual)

def J_integrand(u, vert0, vert1, x):
    grad_phi_i_x = np.gradient(bf.phi_2d(vert0, x, h), axis=0)
    grad_phi_i_y = np.gradient(bf.phi_2d(vert0, x, h), axis=1)
    grad_phi_j_x = np.gradient(bf.phi_2d(vert1, x, h), axis=0)
    grad_phi_j_y = np.gradient(bf.phi_2d(vert1, x, h), axis=1)
    grad_u_x = np.gradient(u, axis=0)
    grad_u_y = np.gradient(u, axis=1)  
    
    summand_1_part_1 = 2*(grad_u_x*grad_phi_i_x + grad_u_y*grad_phi_i_y)
    summand_1_part_2 = np.multiply(grad_u_x, grad_phi_i_x) + np.multiply(grad_u_y, grad_phi_i_y)
    
    summand_1 = summand_1_part_1 * summand_1_part_2
    
    summand_2_part_1 = np.square(grad_u_x) + np.square(grad_u_y) 
    summand_2_part_2 = np.multiply(grad_phi_j_x, grad_phi_i_x) + np.multiply(grad_phi_i_y, grad_phi_i_y)
    
    summand_2 = summand_2_part_1 * summand_2_part_2
        
    return summand_1 + summand_2
    
def test_phi(x):
    temp = 0
    for i, vert in enumerate(vertices):
        temp = temp + bf.phi_2d(vert, x, h)
    return temp
    
import torch
#linear_sol = torch.load("poisson_2d_sol.pt")
u = np.exp(np.multiply(X, Y))
#u = u.reshape(H, H)
u_iterations = [u]
steps = int(H*2)    

"""plt.imshow(u)
#cmax = np.amax(u)
#plt.clim(0, cmax)
plt.colorbar()
plt.pause(.01)"""

t0 = time.time()
max_change_0 = 0
for t in range(100):
    u_interp = scipy.interpolate.interp2d(x, y, u, kind="linear")
    
    for i, vert0 in enumerate(vertices):
        
        # defining integration limits
        x0 = np.linspace(vert0[0]-h, vert0[0]+h, steps)
        y0 = np.linspace(vert0[1]-h, vert0[1]+h, steps)
        X0, Y0 = np.meshgrid(x0, y0) 
        u_slice = u_interp(x0, y0)
        
        integrand0 = f_integrand(u_slice, vert0, (X0, Y0))
        F[i] = np.trapz(np.trapz(integrand0, y0, axis=0), x0, axis=0)
        
        percentage = round(100 * ((i)/(num_vertices-1)), 1)
        time_running = round((time.time() - t0)/60, 1)
        print(f"{percentage}% !! iteration: {t} !! running for {time_running} minutes", end="\r")

        for j, vert1 in enumerate(vertices):#[0:i+1]):
            
            integrand1 = J_integrand(u_slice, vert0, vert1, (X0, Y0))
            J[i, j] = np.trapz(np.trapz(integrand1, y0, axis=0), x0, axis=0)
            #J[j, i] = J[i, j]
            
    u_hat = np.linalg.lstsq(J, F, rcond=None)[0]
    u_hat = u_hat.reshape(H, H)
    
    u = u + u_hat
    u_iterations.append(u)
    
    if t == 0:
        max_change_0 = np.max(np.abs(u - u_hat))
    else:
        max_change = np.max(np.abs(u - u_hat))
        
        if np.abs(max_change_0 - max_change) <= max_change_0*0.05:
            print("done")
            break 
        
    
    #integrand0 = f_integrand(u, [.5, .5], (X, Y))
    #test = np.trapz(np.trapz(integrand0, y0, axis=0), x0, axis=0)
    
    """plt.clf()
    plt.imshow(u)
    #plt.clim(0, cmax)
    plt.colorbar()
    plt.pause(.01)"""

    torch.save(u_iterations, "converging_func.pt")