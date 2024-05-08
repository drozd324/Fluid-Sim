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
x = np.linspace(0, 1, N)   

# vertices to solve
num_vertices = 5
vertices = np.linspace(0, 1, num_vertices)
h = vertices[1] - vertices[0]

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

def basis_func(vert):
    return bf.phi(vert, x, h, boundary=[0, 1])

# main calculation loop
for i in range(num_vertices):
    F[i] = np.trapz(basis_func(vertices[i]) * f(x), dx=h)

    for j in range(num_vertices):

        #A[i, j] = np.trapz(bf.grad_phi(vertices[i], x, h) * bf.grad_phi(vertices[j], x, h), dx=h)
        A[i, j] = np.trapz(np.gradient(basis_func(vertices[i])) * np.gradient(basis_func(vertices[j])), dx=h)


# this is solving the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F, rcond=None) # lqst because this aproximate matrix solution has more than one solution

u = 0
for i, vert in enumerate(vertices):
    u = u + (solution[0][i] * bf.phi(vert, x, h))

#plt.plot(vertices, solution[0])
plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u")
plt.title(r'Solution for Poisson Problem: $u^{\prime\prime} (x) = -1$')

plt.matshow(A)
plt.colorbar()
plt.show()