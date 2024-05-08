import numpy as np
import matplotlib.pyplot as plt

# domain for calculating inetgrals
H = .01
H_len = int(1/H)
#domain = np.mgrid[0: 1: H , 0: 1: H]
N = 100
domain = np.linspace(0, 1, N)

# elements to solve
h = .1
x_len = int(1/h)
num_elements = (x_len)**2
elements0 = np.mgrid[0: 1+h: h, 0: 1+h: h]
elements  = np.mgrid[0: 1+h: h, 0: 1+h: h].reshape(2, -1).T

# 1dim hat basis function
def phi(j, x):
    print(j)
    if j==0 or j==1:# 0,1 are the boundary conditions, ie 
        return np.zeros(x.shape)
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j)        , (x > (j))&(x < (j+h))          , x>=(j+h)],
                               [0         , lambda x: (x/h) + (1-(j/h)) , lambda x: (-x/h) + (1+(j/h))   , 0       ])

# N-dim hat basis function
def phi_Ndim(j, x):
    dim1_phis = []
    dim = len(j)
    for i in range(dim):
        dim1_phis.append(phi(j[i], x[i]))
    
    phis_num = len(dim1_phis)
    for i in range(phis_num):
        dim1_phis[-(i+1)] = np.multiply(dim1_phis[-(i+1)], dim1_phis[-i])
    
    return dim1_phis[0]

def grad_phi(j, x):
    return np.gradient(phi(j, x))

def grad_phi_Ndim(j, x):
    return np.gradient(phi_Ndim(j, x))  

def f(x):
    return -1

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros((num_elements))

# main calculation loop
for i in range(num_elements):
    print(elements[i], i)
    num1 = np.trapz(phi_Ndim(elements[i], domain) * f(domain), dx=h, axis=1)
    num2 = np.trapz(num1, dx=h)
    F[i] = num2

    for j in range(num_elements):
        num1 = np.trapz(grad_phi_Ndim(elements[i], domain) * grad_phi_Ndim(elements[j], domain), dx=h)
        num2 = np.trapz(num1, dx=h)
        A[i, j] = num2

# this is soling the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F) # lqst because this aproximate matrix solution has more than one solution

import plotly.graph_objects as go
fig = go.Figure(go.Surface(
    x = domain,
    y = domain,
    z = solution
    ))
fig.show()