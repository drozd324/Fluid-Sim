import numpy as np
import scipy.integrate
import plotly.graph_objects as go

# domain for calculating integrals
H = 0.01
H_len = int(1 / H)
N = 10
domain = np.linspace(0, 1, N)

# elements to solve
h = 0.2
x_len = int(1 / h)
num_elements = x_len**2
elements = np.mgrid[0:1 + h:h, 0:1 + h:h].reshape(2, -1).T

# 1dim hat basis function
def phi(j, x):
    if j == 0 or j == 1:  # 0,1 are the boundary conditions
        return np.zeros_like(x)
    else:
        return np.piecewise(x, [x <= (j - h), (x > (j - h)) & (x <= j),
                                (x > j) & (x < (j + h)), x >= (j + h)],
                            [0, lambda x: (x / h) + (1 - (j / h)),
                                lambda x: (-x / h) + (1 + (j / h)), 0])

# N-dim hat basis function
def phi_Ndim(j, *x):
    dim1_phis = []
    dim = len(j)
    for i in range(dim):
        dim1_phis.append(phi(j[i], x[i]))

    phis_num = len(dim1_phis)
    for i in range(1, phis_num):
        dim1_phis[-i - 1] = np.multiply(dim1_phis[-i - 1], dim1_phis[-i])

    return dim1_phis[0]

def grad_phi(j, x):
    if j == 0 or j == 1:  # 0,1 are the boundary conditions
        return 0
    else:
        return np.piecewise(x, [x <= (j - h), (x > (j - h)) & (x <= j),
                                (x > j) & (x < (j + h)), x >= (j + h)],
                            [0, 1 / h, -1 / h, 0])

def grad_phi_Ndim(j, *x):
    dim = len(j)
    solution = []

    for a in range(dim):
        grad = grad_phi(j[a], x[a])
        popped_j = list(j)
        popped_j.pop(a)

        for b in range(len(popped_j)):
            grad = grad * phi(popped_j[b], x[b])

        solution.append(grad)

    return np.array(solution)

def f(*x):
    return -1

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros(num_elements)

# main calculation loop
for i in range(num_elements):
    def func1(x, y):
        return phi_Ndim(elements[i], x, y) * f(x, y)

    F[i] = scipy.integrate.nquad(func1, [[0, 1], [0, 1]])[0]

    percentage = round(100 * ((i) / (num_elements - 1)), 1)
    print(f"{percentage}%", end="\r")

    for j in range(num_elements):
        def func2(x, y):
            return np.dot(grad_phi_Ndim(elements[j], x, y), grad_phi_Ndim(elements[i], x, y))

        A[i, j] = scipy.integrate.nquad(func2, [[0, 1], [0, 1]])[0]

# solve the matrix equation Ax = F
solution = np.linalg.lstsq(A, F, rcond=None)
print(solution)

fig = go.Figure(go.Surface(
    x=domain,
    y=domain,
    z=solution[0].reshape(x_len, x_len)
))
fig.show()
fig = go.Figure(go.Surface(
    x=domain,
    y=domain,
    z=solution[-1].reshape(x_len, x_len)
))
fig.show()
