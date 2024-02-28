import numpy as np
import scipy

# domain for calculating inetgrals
H = .1
H_len = int(1/H)
#domain = np.mgrid[0: 1: H , 0: 1: H]
N = 10
domain = np.linspace(0, 1, N)

# elements to solve
h = .25
x_len = int(1/h)
num_elements = (x_len)**2
elements0 = np.mgrid[0: 1+h: h, 0: 1+h: h]
elements  = np.mgrid[0: 1+h: h, 0: 1+h: h].reshape(2, -1).T


#print(elements.shape)
#print(elements)

# 1dim hat basis function
def phi(j, x):
    #print(j)
    if j==0 or j==1:# 0,1 are the boundary conditions
        return 0
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j)        , (x > (j))&(x < (j+h))          , x>=(j+h)],
                               [0         , lambda x: (x/h) + (1-(j/h)) , lambda x: (-x/h) + (1+(j/h))   , 0       ])

# N-dim hat basis function
def phi_Ndim(j, *x):
    dim1_phis = []
    dim = len(j)

    if (j==0).any() == 0 or (j==1).any() == 1:
        return 0
    
    else:
        for i in range(dim):
            dim1_phis.append(phi(j[i], x[i]))
        
        phis_num = len(dim1_phis)
        for i in range(phis_num):
            dim1_phis[-(i+1)] = np.multiply(dim1_phis[-(i+1)], dim1_phis[-i])
        
        return float(dim1_phis[0])

def grad_phi(j, x):
    if j==0 or j==1:# 0,1 are the boundary conditions
        return 0
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j), (x > (j))&(x < (j+h)), x>=(j+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])

# turn this into a constructor thing
def grad_phi_Ndim(j, *x):
    dim = len(j,)
    solution = []
    
    for a in range(dim):
        grad = grad_phi(j[a], x[a])
        popped_j = list(j)
        popped_j.pop(a)
                
        for b in range(len(popped_j)):
            grad = grad * phi(popped_j[b], x[b])
        
        solution.append(float(grad))
    
    return np.array(solution)

def f(*x):
    return x[0]**3

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros((num_elements))


# main calculation loop # the matrix is symmetric make use of that
for i in range(num_elements):
    def func1(x, y):
        return phi_Ndim(elements[i], x, y) * (-1)
    
    #F[i] = scipy.integrate.nquad(func1, [[0, 1], [0, 1]])[0] #these scipy quad function may be too powerful for what we need
    F[i] = scipy.integrate.dblquad(func1, 0, 1, 0, 1, epsabs=1e-3)[0] #these scipy quad function may be too powerful for what we need
    
    percentage = round(100 * ((i)/(num_elements-1)), 1)
    print(f"{percentage}%", end="\r")

    for j in range(num_elements):
        def func2(x, y):
            return np.dot(grad_phi_Ndim(elements[j], x, y), grad_phi_Ndim(elements[i], x, y))
        
        #A[i, j] = scipy.integrate.nquad(func2, [[0, 1], [0, 1]])[0]
        A[i, j] = scipy.integrate.dblquad(func2, 0, 1, 0, 1, epsabs=1e-3)[0]

# this is soling the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F, rcond=None) # lqst because this aproximate matrix solution has more than one solution
print(solution)

import matplotlib.pyplot as plt
plt.matshow(A)
plt.colorbar()
plt.show()

import plotly.graph_objects as go
fig = go.Figure(go.Surface(
    x = domain,
    y = domain,
    z = solution[0].reshape(x_len,x_len)
    ))
fig.show()
fig = go.Figure(go.Surface(
    x = domain,
    y = domain,
    z = solution[-1].reshape(x_len,x_len)
    ))
fig.show()