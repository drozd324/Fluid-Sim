


import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

N = 200

def solve_poisson(H):    
    """Container function for main part of code. The purpose of this is to create an easy way to 
    run solver with various parameters.

    Args:
        H (int): number of nodes

    Returns:
        _type_: _description_
    """
    
    #H = 50 # number of vertices
    vertices = list(np.linspace(0, 1, H))[:-1]
    H = H-1
    elements = [[vertices[i], vertices[i+1]] for i in range(H-1)]
    h = vertices[1] - vertices[0]
    precision = 0.0001 # precision in integration in each element
    num_step = round(1/precision)
    #print(num_step)

    def force(x):
        return -1

    def neum(x):
        return -x  #x - (1/2)#-2*x

    # matrices to compute
    A = np.zeros((H, H))
    F = np.zeros((H))

    def basis_func(vert, x):    
        #if vert == 0:
        #    return bf.hat(0, x, h) + bf.hat(1, x, h) #0*x
        #else:
        #    return bf.hat(vert, x, h)
        return bf.phi(vert, x, h)

    def grad_basis_func(vert, x):
        #if vert == 0:
        #    return bf.grad_hat(0, x, h) + bf.grad_hat(1, x, h) #0*x
        #else:
        #   return bf.grad_hat(vert, x, h)
        return bf.grad_phi(vert, x, h)

    def a(vert0, vert1, x):
        return grad_basis_func(vert0, x) * grad_basis_func(vert1, x)

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
    
    A_pbc = A
    #A_pbc[0, 1:-1] = A[0   , 1:-1] + A[-1  , 1:-1]
    #A_pbc[1:-1, 0] = A[1:-1, 0   ] + A[1:-1, -1  ]
    
    A_pbc[0, :] = A[0, :] + A[-1, :]
    A_pbc[:, 0] = A[:, 0] + A[: , -1]
    
    #A_pbc[-1, 1:-1] = A[0   , 1:-1] + A[-1  , 1:-1]
    #A_pbc[1:-1, -1] = A[1:-1, 0   ] + A[1:-1, -1  ]
    A_pbc[0, 0] = A[0, 0] + A[0, -1] + A[-1, 0] + A[-1, -1]
    
    F_pbc = F
    F_pbc[0] = F[0] + F[-1]
    
    # trim last entries
    A_pbc = A_pbc[0:-1, 0:-1]
    F_pbc = F_pbc[0:-1]
    
    solution_pbc = np.matmul(np.linalg.pinv(A_pbc), F_pbc)
    print(32)
    
    solution_pbc = np.array((list(solution_pbc).append(solution_pbc[0])))

    x = np.linspace(0, 1, N)
    u = bf.conv_sol(solution_pbc, x, bf.hat, vertices, h)
    
    return u, A_pbc, F_pbc

poisson_sols = []
vert_num = [1000]
for num in vert_num:
    sol = solve_poisson(num)
    poisson_sols.append(sol)

period = 3
x = np.linspace(0, period, N*period)[:-period]
#plt.plot(x, ((x**2)/2) - (x/2), label="Analytical Solution")
#plt.plot(x, x**2, label="Analytical Solution")
#plt.plot(x, (-(x**2)/2) + x - (1/4), label="Analytical Solution")
for i, u in enumerate(poisson_sols):
    plt.plot(np.block(x), np.block(([u[0][:-1], u[0][:-1], u[0][:-1]])), label=f"{vert_num[i]} vertices")
    #plt.plot(x, u[0], label=f"{vert_num[i]} vertices")
    plt.matshow(u[1])
    plt.show()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
#plt.title(r'Solution for $u^{\prime\prime}(x) = -1$')
plt.savefig(f"(Poisson_1d)_(lin_sol)")

print(len(poisson_sols[0][0]))
x = np.linspace(0, 1, N)
print(np.gradient(np.gradient(poisson_sols[0][0], x), x) - (-1))


#plt.matshow(A)
#plt.title(r'Solution Matrix for $u^{\prime\prime}(x) = -1$')
#plt.colorbar()
#plt.savefig(f"(Poisson_1d)_(lin_matrix_A)_(vertex_num_{H})")

#F = [[F[i], 0] for i in range(H)]
#plt.matshow(F)
#plt.colorbar()

#plt.show()  