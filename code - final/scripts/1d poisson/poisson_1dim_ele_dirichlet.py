"""
This script solves the 1 dimensional poisson probelem with dirchlet boudary conditions. Section 3 in report
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf
from tools import norms as nrm

N = 200
num_step = 10000 # number of steps in calculation of integrals

def solve_poisson(H):   
    """Container function for main part of code. The purpose of this is to create an easy way to 
    run solver with various parameters.

    Args:
        H (int): number of nodes

    Returns:
        tuple: solution to equation, solution matrix A, solution matrix F
    """ 
    
    vertices = list(np.linspace(0, 1, H))
    elements = [[vertices[i], vertices[i+1]] for i in range(H-1)]
    h = vertices[1] - vertices[0]

    #defining basis functions and their derivatives. This was done initially to swap out linear basis functions to quadratic basis function just because we can
    def basis_func(vert, x):
        return bf.phi(vert, x, h)

    def d_basis_func(vert, x):
        return bf.grad_phi(vert, x, h)

    def force(x):
        return -1

    #defining matrix to calculate entries for
    A = np.zeros((H, H))
    F = np.zeros((H))
    
    # functions for calculating entries of matrices
    def a(vert0, vert1, x):
        return d_basis_func(vert0, x) * d_basis_func(vert1, x)

    def f(vert0, x):
        return basis_func(vert0, x) * force(x)

    # main bit of code iterating over elements of mesh (here a 1dim line) and calcualting entries of the matrices
    for ele in elements:
        x = np.linspace(ele[0], ele[1], num_step)
        
        for vert0 in ele:
            j = vertices.index(vert0)
            F[j] = F[j] + np.trapz(f(vert0, x), x)
            
            for vert1 in ele:
                i = vertices.index(vert1)
                A[i, j] = A[i, j] + np.trapz(a(vert0, vert1, x), x)

    # final calculation of solution
    A_inverted = np.linalg.pinv(A)
    solution = np.matmul(A_inverted, F)
    
    x = np.linspace(0, 1, N)
    u = bf.conv_sol(solution, x, bf.hat, vertices, h)
    
    return u, A, F



poisson_sols = []
vert_num = [3, 5, 20] # number of vertices to solve the equation over
for num in vert_num:
    sol = solve_poisson(num)
    poisson_sols.append(sol)

x = np.linspace(0, 1, N)
for i, u in enumerate(poisson_sols):
    plt.plot(x, u[0], label=f"{vert_num[i]} vertices")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x, ((x**2)/2) - (x/2), label="Analytical Solution")
plt.legend()
plt.title(r'Solution for $-u^{\prime\prime}(x) = 1$')
#plt.savefig(f"(Poisson_1d)_(lin_sol)_(x)", dpi=500)
plt.show()  

for i in range(len(vert_num)):
    plt.matshow(poisson_sols[i][1])
    plt.title(r'Solution Matrix for $-u^{\prime\prime}(x) = 1$' + f"   ,vert num = {vert_num[i]}")
    plt.colorbar()
    plt.show()
    #plt.savefig(f"(Poisson_1d)_(lin_matrix_A)_(vertex_num_{H})")
    
""" a method of measuring how accurate our solutions are would be to use a norm on functions.
There is an $L^2$ norm which is used to comapare functions. We would simply take the difference 
of our solution and its analytical solution and throw it into this norm.
"""

print()