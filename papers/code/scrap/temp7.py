import numpy as np
import scipy.integrate
import plotly.graph_objects as go

# domain for calculating integrals
H = 0.01
N = 10
domain = np.linspace(0, 1, N)

# elements to solve
h = 0.25
x_len = int(1 / h)
num_elements = x_len ** 2
elements = np.mgrid[0:1 + h:h, 0:1 + h:h].reshape(2, -1).T

# 1dim hat basis function
def phi(j, x):
    return np.piecewise(x,
                        [x <= (j - h), (x > (j - h)) & (x <= j), (x > j) & (x < (j + h)), x >= (j + h)],
                        [0, lambda x: (x / h) + (1 - (j / h)), lambda x: (-x / h) + (1 + (j / h)), 0])

# N-dim hat basis function
def phi_Ndim(j, *x):
    dim1_phis = np.array([phi(j[i], x[i]) for i in range(len(j))])
    return np.prod(dim1_phis, axis=0)

def grad_phi(j, x):
    return np.piecewise(x,
                        [x <= (j - h), (x > (j - h)) & (x <= j), (x > j) & (x < (j + h)), x >= (j + h)],
                        [0, 1 / h, -1 / h, 0])

def grad_phi_Ndim(j, *x):
    dim = len(j)
    solution = np.ones_like(x[0])

    grad = np.array([grad_phi(j[a], x[a]) for a in range(dim)])
    phi_values = np.array([phi(j[b], x[b]) for b in range(dim)])

    for a in range(dim):
        solution *= np.prod(np.delete(phi_values, a, axis=0), axis=0) * grad[a]

    return solution


def f(*x):
    return -1

# matrices to solve
A = np.zeros((num_elements, num_elements))
F = np.zeros(num_elements)

# main calculation loop
for i in range(num_elements):
    func1 = lambda x, y: phi_Ndim(elements[i], x, y) * f(x, y)
    F[i] = scipy.integrate.nquad(func1, [[0, 1], [0, 1]])[0]

    percentage = round(100 * ((i) / (num_elements - 1)), 1)
    print(f"{percentage}%", end="\r")

    for j in range(num_elements):
        func2 = lambda x, y: np.dot(grad_phi_Ndim(elements[j], x, y), grad_phi_Ndim(elements[i], x, y))
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
