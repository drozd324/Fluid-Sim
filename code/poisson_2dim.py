import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# domain for calculating integrals
N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
#print(X.shape)
#print(Y.shape)

# elements to solve
H = 5
h = 1/H
a = np.linspace(0, 1, H)
b = np.linspace(0, 1, H)
num_elements = H**2
elements = np.array(np.meshgrid(a, b)).reshape(2, -1).T

# 1dim hat basis function
def phi(i, x):
    #print(i)
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i)        , (x > (i))&(x < (i+h))          , x>=(i+h)],
                               [0         , lambda x: (x/h) + (1-(i/h)) , lambda x: (-x/h) + (1+(i/h))   , 0       ])

# 2dim hat basis function
def phi_2d(i, x):
    return phi(i[0], x[0]) * phi(i[1], x[1])


def grad_phi(i, x):
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i), (x > (i))&(x < (i+h)), x>=(i+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])
        
def grad_phi_2d(i, x):
    return np.array([grad_phi(i[0], x[0]) * phi(i[1], x[1]), phi(i[0], x[0]) * grad_phi(i[1], x[1])])

def f(x):
    return -1

A = np.zeros((num_elements, num_elements))
print(A.shape)
F = np.zeros((num_elements))

def func1(i, x):
    return phi_2d(i, x) * f(x)

def func2(i, a, x):
    return grad_phi(i[0], x[0]) * phi(i[1], x[1]) * grad_phi(a[0], x[0]) * phi(a[1], x[1])  +  phi(i[0], x[0]) * grad_phi(i[1], x[1]) * phi(a[0], x[0]) * grad_phi(a[1], x[1])

#def func2(i, a, x):
#    return grad_phi_2d(i, x) * grad_phi_2d(a, x)

# main calculation loop 
# to optimise: make use of symmetric matrix fact
#              sparse matrix solving?
#              calculate integral only on where the integrand is defined               
for i, ele0 in enumerate(elements):
    integrand0 = func1(ele0, (X, Y))
    F[i] = np.trapz(np.trapz(integrand0, y, axis=0), x, axis=0)
    
    percentage = round(100 * ((i)/(num_elements-1)), 1)
    print(f"{percentage}%", end="\r")

    for j, ele1 in enumerate(elements):
        integrand1 = func2(ele0, ele1, (X, Y))
        A[i, j] = np.trapz(np.trapz(integrand1, y, axis=0), x, axis=0)

# this is soling the matrix equation Ax = F, A a matrix, x a vector ie the solution at each element, F a vector
solution = np.linalg.lstsq(A, F, rcond=None)

plt.matshow(A)
plt.colorbar()
#plt.show()

fig = go.Figure(go.Surface(
    x = x,
    y = y,
    z = solution[0].reshape(H, H)
    ))
fig.show()