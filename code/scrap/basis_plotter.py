import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# domain for calculating inetgrals
H = .01
H_len = int(1/H)
domain = np.mgrid[0: 1: H , 0: 1: H]

# elements to solve
h = .1
x_len = int(1/h)
num_elements = (x_len)**2
elements0 = np.mgrid[0: 1  : h, 0: 1  : h]
elements  = np.mgrid[0: 1+h: h, 0: 1+h: h].reshape(2, -1).T

# 1dim hat basis function
def phi(j, x):
    if j==0 or j==1:# 0,1 are the boundary conditions, ie 
        return np.zeros((x,).shape)
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j)        , (x > (j))&(x < (j+h))          , x>=(j+h)],
                               [0         , lambda x: (x/h) + (1-(j/h)) , lambda x: (-x/h) + (1+(j/h))   , 0       ])
    
# 2dim hat basis function
def phi_2d(j, x):
    j0, j1 = j
    x0, x1 = x
    return np.multiply(phi(j0, x0), phi(j1, x1))

# n-dim hat basis function
def phi_Ndim(j, x):
    dim1_phis = []
    dim = len(j) # = len(j)
    for i in range(dim):
        dim1_phis.append(phi(j[i], x[i]))
    
    phis_num = len(dim1_phis)
    for i in range(phis_num):
        dim1_phis[-(i+1)] = np.multiply(dim1_phis[-(i+1)], dim1_phis[-i])
    
    return dim1_phis[0]


"""temp = phi_Ndim((.5, .3), domain)
fig = go.Figure(go.Surface(
    x = np.linspace(0, 1, H_len),
    y = np.linspace(0, 1, H_len),
    z = temp
    ))
fig.show()
"""

def grad_phi(j, x):
    if j==0 or j==1:# 0,1 are the boundary conditions, ie 
        return 0
    else:
        return np.piecewise(x, [x <= (j-h), (x > (j-h))&(x <= j), (x > (j))&(x < (j+h)), x>=(j+h)],
                               [0         , lambda x: 1/h       , lambda x: -1/h       , 0       ])
    
def grad_phi_Ndim(j, *x):
    dim = len(j,)
    solution = []
    
    for a in range(dim):
        grad = grad_phi(j[a], x[0][a])
        popped_j = list(j)
        popped_j.pop(a)
        #print(len(popped_j))
        
        for b in range(len(popped_j)):
            grad *= phi(popped_j[b], x[0][b])
        
        solution.append(grad)
    
    return np.array(solution)

def grad_phi_2d(i, x):
    return np.array([grad_phi(i[0], x[0]) * phi(i[1], x[1]), phi(i[0], x[0]) * grad_phi(i[1], x[1])])

def hat(i, x, h):
    return np.piecewise(x, [np.abs(x-i) < 1            , np.abs(x-i) >= 1],
                           [lambda x: 1 - np.abs(x-i)  , 0             ])


x = np.linspace(-10, 10, 1000)
#plt.plot(x, grad_phi(.5, x))
#plt.plot(x, phi(.5, x) + 4*phi(.6, x))
plt.plot(x, hat(.5 ,x, 2))
plt.show()

"""temp = grad_phi_Ndim((.5, .3), domain)
print(domain.shape)
print(temp)
fig = go.Figure(go.Surface(
    x = np.linspace(0, 1, H_len),
    y = np.linspace(0, 1, H_len),
    z = temp[0]
    ))
fig.show()
"""