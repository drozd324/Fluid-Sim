import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

# domain for calculating inetgrals
N = 200
domain = np.linspace(0, 1, N)   

# vertices to solve
num_vertices = 20
vertices = np.linspace(0, 1, num_vertices)
h = vertices[1] - vertices[0]

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

# main calculation loop
for i in range(num_vertices):
    F[i] = np.trapz(bf.phi(vertices[i], domain, h) * f(domain), dx=h)

    for j in range(num_vertices):

        A[i, j] = np.trapz(bf.grad_phi(vertices[i], domain, h) * bf.grad_phi(vertices[j], domain, h), dx=h)


# this is solving the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F) # lqst because this aproximate matrix solution has more than one solution

plt.plot(vertices, solution[0])
plt.xlabel("x")
plt.ylabel("u")
plt.title(r'Solution for Poisson Problem: $u^{\prime\prime} (x) = -1$')


plt.matshow(A)
plt.colorbar()
plt.show()