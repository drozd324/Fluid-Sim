import numpy as np
import matplotlib.pyplot as plt

# domain for calculating inetgrals
N = 100
domain = np.linspace(0, 1, N)

# elements to solve
h = .1
x_len = int(1/h)
num_elements = (x_len)**2
elements = np.mgrid[0: 1+h: h, 0: 1+h: h].reshape(2, -1).T

# 1dim hat basis function
def phi(j, x):
    if j==0 or j==1:# 0,1 are the boundary conditions, ie 
        return np.zeros(x.shape)
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j)        , (x > (j))&(x < (j+h))          , x>=(j+h)],
                               [0         , lambda x: (x/h) + (1-(j/h)) , lambda x: (-x/h) + (1+(j/h))   , 0       ])
    
# 2dim hat basis function
def phi_2d(j, x):
    j0, j1 = j
    x0, x1 = x
    return phi(j0, x0) * phi(j1, x1)

def grad_phi(j, x):
    return np.gradient(phi(j, x))

def grad_phi_2d(j, x):
    return np.gradient(phi_2d(j, x))

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros((num_elements))

# main calculation loop
for i in range(num_elements):
    print(elements[i])
    num1 = np.trapz(phi_2d(elements[i], domain) * f(domain), dx=h)
    num2 = np.trapz(num1, dx=h)
    F[i] = num2

    for j in range(num_elements):
        num1 = np.trapz(grad_phi_2d(elements[i], domain) * grad_phi_2d(elements[j], domain), dx=h)
        num2 = np.trapz(num1, dx=h)
        A[i, j] = num2

# this is soling the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F) # lqst because this aproximate matrix solution has more than one solution

#plt.xlabel("x")
#plt.ylabel("u")
#plt.title(r'$ \nabla{^2u(x)} = u^{\prime\prime} (x) = f(x)$')
plt.imshow(solution.reshape(num_elements, num_elements))
plt.show()