import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def mesh(xs, ys, npoints):
    # randomly choose some points
    rng = np.random.default_rng(1234567890)
    rx = rng.uniform(xs[0], xs[1], size=npoints)
    ry = rng.uniform(ys[0], ys[1], size=npoints)
    # only take points in domain
    points = np.column_stack((rx, ry))
    domain_points = points[in_domain(points[:,0], points[:,1])]
    # Delaunay triangulation
    tri = Delaunay(domain_points)
    return tri
            
def A_e(v):
    # take vertices of element and return contribution to A
    G = np.linalg.inv(np.vstack((np.ones((1,3)), v.T))) @ np.vstack((np.zeros((1,2)), np.eye(2)))
    return np.linalg.det(np.vstack((np.ones((1,3)), v.T))) * G @ G.T / 2

def b_e(v):
    # take vertices of element and return contribution to b
    vS = np.sum(v, axis=0) / 3.0 # Centre of gravity
    return f(vS) * ((v[1,0]-v[0,0])*(v[2,1]-v[0,1])-(v[2,0]-v[0,0])*(v[1,1]-v[0,1])) / 6.0

def poisson(tri):
    # get elements and vertices from mesh
    elements = tri.simplices
    vertices = tri.points
    # number of vertices and elements
    N = vertices.shape[0]
    E = elements.shape[0]
    #Loop over elements and assemble LHS and RHS 
    A = np.zeros((N,N))
    b = np.zeros((N,1))
    for j in range(E):
        index = elements[j,:]
        A[np.ix_(index,index)] += A_e(vertices[index,:])
        b[index] += b_e(vertices[index,:])
    # find the boundary vertices
    boundary = np.unique(np.concatenate((tri.convex_hull, np.roll(tri.convex_hull, -1, axis=1))))
    # find the 'free' vertices that we need to solve for    
    free = list(set(range(len(vertices))) - set(boundary))
    # initialise solution to zero so 'non-free' vertices are by default zero
    u = np.zeros((N,1))
    # solve for 'free' vertices.
    u[free] = np.linalg.solve(A[np.ix_(free,free)],b[free])
    return u

def f(v):
    # the RHS f
    x, y = v
    return 1 #2.0 * np.cos(10.0 * x) * np.sin(10.0 * y) + np.sin(10.0 * x * y)

def in_domain(x, y):
    # is a point in the domain?
    return np.sqrt(x**2 + y**2) <= 1

xs = (-1.,1.)
ys = (-1.,1.)
npoints = 1000

# generate mesh
tri = mesh(xs, ys, npoints)

# solve Poisson equation
u  = poisson(tri)

# interpolate values and plot a nice image
x_vals = np.linspace(xs[0], xs[1], npoints)
y_vals = np.linspace(ys[0], ys[1], npoints)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)
for i in range(npoints):
    for j in range(npoints):
        Z[i,j] = f([X[i,j], Y[i,j]])

plt.figure()
plt.imshow(Z, interpolation='bilinear', extent=(xs[0], xs[1], ys[0], ys[1]), origin='lower')
plt.colorbar()
plt.savefig('sol.png', bbox_inches='tight')
plt.show()
