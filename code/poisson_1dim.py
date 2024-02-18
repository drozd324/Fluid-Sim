import numpy as np
import matplotlib.pyplot as plt

# domain for calculating inetgrals
N = 100
domain = np.linspace(0, 1, N)   

# vertices to solve
num_vertices = 10
vertices = np.linspace(0, 1, num_vertices)
h = vertices[1] - vertices[0]

def phi(i, x):
    """hat function with peak at i

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i)        , (x > (i))&(x < (i+h))          , x>=(i+h)],
                               [0         , lambda x: (x/h) + (1-(i/h)) , lambda x: (-x/h) + (1+(i/h))   , 0       ])

def grad_phi(i, x):
    """derivative of hat function with peak at i 

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i), (x > (i))&(x < (i+h)), x>=(i+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_vertices, num_vertices))
F = np.zeros((num_vertices))

# main calculation loop
for i in range(num_vertices):
    F[i] = np.trapz(phi(vertices[i], domain) * f(domain), dx=h)

    for j in range(num_vertices):

        A[i, j] = np.trapz(grad_phi(vertices[i], domain) * grad_phi(vertices[j], domain), dx=h)

# this is solving the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F) # lqst because this aproximate matrix solution has more than one solution

plt.plot(vertices, solution[0])
plt.xlabel("x")
plt.ylabel("u")
plt.title(r'Solution for Poisson Problem: $ \nabla{^2u(x)} = u^{\prime\prime} (x) = f(x) = -1$')


plt.matshow(A)
plt.colorbar()
plt.show()