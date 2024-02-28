import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time 

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

# defining mesh
H = 10
h = 1/H
a = np.linspace(0, 1, H)
b = np.linspace(0, 1, H)
num_vertices = H**2
vertices = np.array(np.meshgrid(a, b)).reshape(2, -1).T

def f(x):
    """forcing function

    Args:
        x (tuple): variable

    Returns:
        float:
    """
    return -1

A = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

# preparing functions for later
def func1(i, x):
    return bf.phi_2d(i, x, h) * f(x)

def func2(i, a, x):
    summand1 = bf.grad_phi(i[0], x[0], h) * bf.phi(i[1], x[1], h) * bf.grad_phi(a[0], x[0], h) * bf.phi(a[1], x[1], h)
    summand2 = bf.phi(i[0], x[0], h) * bf.grad_phi(i[1], x[1], h) * bf.phi(a[0], x[0], h) * bf.grad_phi(a[1], x[1], h)
    
    return summand1 + summand2

t0 = time.time()
# main calculation loop 
for i, ele0 in enumerate(vertices):
    
    # defining integration limits
    steps = int(H*2)
    x0 = np.linspace(ele0[0]-h, ele0[0]+h, steps)
    y0 = np.linspace(ele0[1]-h, ele0[1]+h, steps)
    X0, Y0 = np.meshgrid(x0, y0) 
    
    integrand0 = func1(ele0, (X0, Y0))
    F[i] = np.trapz(np.trapz(integrand0, y0, axis=0), x0, axis=0)
    
    percentage = round(100 * ((i)/(num_vertices-1)), 1)
    print(f"{percentage}% non linear haha", end="\r")

    for j, ele1 in enumerate(vertices[0:i+1]):
        
        integrand1 = func2(ele0, ele1, (X0, Y0))
        A[i, j] = np.trapz(np.trapz(integrand1, y0, axis=0), x0, axis=0)
        A[j, i] = A[i, j]
        
t1 = time.time()
print(f"time taken: {(t1-t0)/60} minutes")

# this is soling the matrix equation Ax = F
solution = np.linalg.lstsq(A, F, rcond=None)[0]

# defining the solution u
def u(x, y):
    ans = 0
    for i, ele in enumerate(vertices):
        ans = ans + (solution[i] * bf.phi_2d(ele, (x, y), h))
    return ans

# the rest is plotting

x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y) 

fig = go.Figure(go.Surface(
    x = x,
    y = y,
    z = u(X, Y)
    ))
fig.show()

plt.matshow(A)
plt.colorbar()
plt.show()

import torch

torch.save(solution, "poisson_2d_sol.pt")