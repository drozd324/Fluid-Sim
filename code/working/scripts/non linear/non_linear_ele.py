import numpy as np
import time 
import matplotlib.pyplot as plt
import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

steps = 10 # accuracy or steps in integrals
epsilon = 1 # scaling for iteration
precision = 1e-3 # criteria for stopping of iterator

# defining mesh
H = 10
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
h = x[1] - x[0]
vertices = (np.array(np.meshgrid(x, y)).reshape(2, -1).T).tolist()
elements = [[[x[i]  , y[j]  ],
             [x[i+1], y[j]  ],
             [x[i]  , y[j+1]],
             [x[i+1], y[j+1]]] for i in range(H-1) for j in range(H-1)]
X, Y = np.meshgrid(x, y)

u = np.array([ 0 for x in vertices])
#u = np.array([ np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1]) for x in vertices])
#u = torch.load("non_linear_ele_sols.pt")[-1].reshape(49)
u_iter = [u]

def force(x):
    # for solution u = sin(pi*x)sin(pi*y)
    x, y = x
    
    term1 = (np.cos(np.pi*x)**2) * (np.sin(np.pi*y)**2)
    term2 = (np.sin(np.pi*x)**2) * (np.cos(np.pi*y)**2)
    term3 = - 2 * (np.sin(np.pi*x)**2) * (np.sin(np.pi*y)**2)
    term4 = - 2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    
    return -(np.pi**2)*(term1 + term2 + term3 + term4)
    

def f_integrand(u, vert, X0):
    dy_u, dx_u = np.gradient(u, h, edge_order=2)
    
    #dx_u, dy_u = np.gradient(u, h) # gradient maybe not correct
    
    grad_dot_grad = (dx_u * bf.dx_phi_2d(vert, X0, h)) + (dy_u * bf.dy_phi_2d(vert, X0, h))
    
    return ((u + 1)*grad_dot_grad) - (force(X0) * bf.phi_2d(vert, X0, h))

def J_integrand(u, vert0, vert1, X0):
    dy_u, dx_u = np.gradient(u, h, edge_order=2)

    term1 = bf.dx_phi_2d(vert0, X0, h) * ((bf.phi_2d(vert1, X0, h) * dx_u) + ((u + 1) * bf.dx_phi_2d(vert1, X0, h)))
    term2 = bf.dy_phi_2d(vert0, X0, h) * ((bf.phi_2d(vert1, X0, h) * dy_u) + ((u + 1) * bf.dy_phi_2d(vert1, X0, h)))

    return term1 + term2


def l_squ_norm(func, x):
    return np.sqrt( np.trapz(np.trapz( func**2 , x, axis=0), x, axis=0) )

t0 = time.time()
#run = True
#t = -1
#while run:
#    t += 1
max_iterations = 500
for t in range(max_iterations):
    J = np.zeros((H**2, H**2))
    F = np.zeros((H**2))

    for k, ele in enumerate(elements):
        x0 = np.linspace(ele[0][0], ele[3][0], steps)
        y0 = np.linspace(ele[0][1], ele[3][1], steps)
        X0, Y0 = np.meshgrid(x0, y0)
        
        u_t = bf.conv_sol(u_iter[t], (X0, Y0), bf.phi_2d, vertices, h)
        
        percentage = round(100 * ((k)/(len(elements)-1)), 1)
        print(f"time {t} and {percentage}%", end="\r")

        for vert0 in ele:
            j = vertices.index(vert0)
            if (np.array(vert0) == 0).any() or (np.array(vert0) == 1).any():
                F[j] = F[j] + bf.conv_sol(u_iter[t], (vert0[0], vert0[1]), bf.hat_2d, vertices, h) #- 0
            else:
                F[j] = F[j] + np.trapz(np.trapz( f_integrand(u_t, vert0, (X0, Y0)) , y0, axis=0), x0, axis=0)
            
            for vert1 in ele:
                i = vertices.index(vert1)
                if (np.array(vert0) == 0).any() or (np.array(vert1) == 1).any() or (np.array(vert1) == 0).any() or (np.array(vert1) == 1).any():
                    J[j, i] = J[j, i] + bf.k_delta(i, j)
                else:
                    J[j, i] = J[j, i] + np.trapz(np.trapz( J_integrand(u_t, vert0, vert1, (X0, Y0)) , y0, axis=0), x0, axis=0)
    
    u_hat = np.matmul(np.linalg.pinv(J), -F)
    
    u = u + (u_hat*epsilon)
    u_iter.append(u)
    
    func_u_sols = []
    for i in range(len(u_iter)):
        func_u_sols.append(np.array(u_iter[i]).reshape(H, H))

    if (t+1) % 500 == 0:
        torch.save(func_u_sols, f"non_linear_big{t}.pt")
    torch.save(func_u_sols, f"non_linear.pt")

    #print(F, np.sqrt(np.sum(np.square(F))))

    print(l_squ_norm(u_hat.reshape(H, H), x))
    print(np.sum(u_hat))
    if l_squ_norm(u_hat.reshape(H, H), x) < precision:
        break
        #run = False
    #elif max_iterations == t:
    #    run = False
        
    #energy_i = energy((u_iter[t+1]).reshape(H, H), (X, Y))
    #energy_terms.append(energy_i)
    #torch.save(energy_terms, "non_linear_ele_energy.pt")
    #print(energy_i)
    
    #plt.matshow(J)
    #plt.colorbar()
    #plt.show()
    
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")