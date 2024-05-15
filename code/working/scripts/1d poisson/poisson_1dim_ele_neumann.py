"""
This script solves the 1 dimensional poisson probelem with neumann boudary conditions.
"""


import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

N = 200
num_step = 10000

def solve_poisson(H):    
    """Container function for main part of code. The purpose of this is to create an easy way to 
    run solver with various parameters.

    Args:
        H (int): number of nodes

    Returns:
        _type_: _description_
    """ 
    
    vertices = list(np.linspace(0, 1, H))
    print(vertices)
    elements = [[vertices[i], vertices[i+1]] for i in range(H-1)]
    h = vertices[1] - vertices[0]

    def force(x):
        return -2

    def neum(x):
        return -2*x

    # matrices to compute
    A = np.zeros((H, H))
    F = np.zeros((H))

    def basis_func(vert, x):
        if vert == 0:
            return 0*x
        else:
            return bf.hat(vert, x, h)

    def d_basis_func(vert, x):
        if vert == 0:
            return 0*x
        else:
            return bf.grad_hat(vert, x, h)
        
    def a(vert0, vert1, x):
        return d_basis_func(vert0, x) * d_basis_func(vert1, x)

    def f(vert0, x):
        return basis_func(vert0, x) * force(x)

    def nuemann(vert0):
        temp = lambda x0: neum(x0) * basis_func(vert0, x0)
        return temp(1) - temp(0)

    for ele in elements:
        x = np.linspace(ele[0], ele[1], num_step)
        
        for vert0 in ele:
            j = vertices.index(vert0)
            F[j] = F[j] + np.trapz(f(vert0, x), x) - nuemann(vert0)
            
            for vert1 in ele:
                i = vertices.index(vert1)
                A[i, j] = A[i, j] + np.trapz(a(vert0, vert1, x), x)


    solution = np.matmul(np.linalg.pinv(A), F)
    #solution_pbc = np.matmul(np.linalg.pinv(A), F)
    
    x = np.linspace(0, 1, N)
    u = bf.conv_sol(solution, x, bf.hat, vertices, h)
    #u = bf.conv_sol(solution_pbc, x, bf.hat, vertices, h)
    
    return u, A, F

poisson_sols = []
vert_num = [5, 20]
for num in vert_num:
    sol = solve_poisson(num)
    poisson_sols.append(sol)

x = np.linspace(0, 1, N)
plt.plot(x, x**2, label="Analytical Solution")
for i, u in enumerate(poisson_sols):
    plt.plot(x, u[0], label=f"{vert_num[i]} vertices", linestyle="--")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title(r'Solution for $u^{\prime\prime}(x) = -1$')
#plt.savefig(f"(Poisson_1d)_(neumann)")

plt.matshow(poisson_sols[-1][1])
plt.colorbar()
#plt.savefig(f"(Poisson_1d)_(lin_matrix_A)_(vertex_num_{H})")

plt.show()  