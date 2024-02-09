import numpy as np
import matplotlib.pyplot as plt

# domain for calculating inetgrals
N = 100
domain = np.linspace(0, 1, N)

# elements to solve
num_elements = 3 # -1
elements = np.linspace(0, 1, num_elements)
h = elements[1] - elements[0]

# 1dim hat basis function
def phi(j, x):
    if j==0 or j==1: # 0,1 are the boundary conditions
        return np.zeros(len(x))
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j)        , (x > (j))&(x < (j+h))          , x>=(j+h)],
                           [0         , lambda x: (x/h) + (1-(j/h)) , lambda x: (-x/h) + (1+(j/h))   , 0       ])

def grad_phi(j, x):
    return np.gradient(phi(j, x))

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros((num_elements))

# main calculation loop
for i in range(num_elements):
    F[i] = np.trapz(phi(elements[i], domain) * f(domain), dx=h)

    for j in range(num_elements):

        A[i, j] = np.trapz(grad_phi(elements[i], domain) * grad_phi(elements[j], domain), dx=h)

# this is soling the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F) # lqst because this aproximate matrix solution has more than one solution

plt.plot(elements, solution[0])
plt.xlabel("x")
plt.ylabel("u")
plt.title(r'Solution for Poisson Problem: $ \nabla{^2u(x)} = u^{\prime\prime} (x) = f(x) = -1$')
plt.show()