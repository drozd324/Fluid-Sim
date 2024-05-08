import numpy as np

"""-------------------- LINEAR BASIS FUNCTIONS -----------------------"""

epsilon = 1e-3

def hat(i, x, h):
    """hat function with peak at i.

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float:         
    """
    return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i)        , (x > (i))&(x < (i+h))           , x>=(i+h)],
                           [0         , lambda x: (x/h) + (1-(i/h)) , lambda x: (-x/h) + (1+(i/h))    , 0                 ])


def grad_hat(i, x, h):
    """derivative of hat function with peak at i 

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i), (x > (i))&(x < (i+h)), x>=(i+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])    


def hat_2d(i, x, h):
    return hat(i[0], x[0], h) * hat(i[1], x[1], h)

def phi_and_hat_2d(i, x, h, hat_bdry=[]):
    return phi(i[0], x[0], h) * phi(i[1], x[1], h, hat_bdry)


def phi(i, x, h, boundary=[0, 1]):
    """Piecewise linear basis function

    Args:
        i (float): position of peak
        x (float): variable
        boundary (tuple): boundary conditions on 

    Returns:
        float: 
    """
    
    if (i == np.array(boundary)).any(): #np.abs(i - 0) < epsilon or np.abs(i - 1) < epsilon: 
        return 0*x
    else:
        return hat(i, x, h)
    
def grad_phi(i, x, h, boundary=[0, 1]):
    """derivative of hat function with peak at i 

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    if (i == np.array(boundary)).any(): #np.abs(i - 0) < epsilon or np.abs(i - 1) < epsilon: 
        return 0*x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i), (x > (i))&(x < (i+h)), x>=(i+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])    
    
def phi_2d(i, x, h):
    """2 dimensional hat function with peak at i.

    Args:
        i (tuple): position of peak
        x (tuple): variable

    Returns:
        float: 
    """

    return phi(i[0], x[0], h) * phi(i[1], x[1], h)

# some derivatives of previously declared functions
def dx_phi_2d(i, x, h):
    return grad_phi(i[0], x[0], h) * phi(i[1], x[1], h)

def dy_phi_2d(i, x, h):
    return phi(i[0], x[0], h) * grad_phi(i[1], x[1], h)

def dx_hat_2d(i, x, h):
    return grad_hat(i[0], x[0], h) * hat(i[1], x[1], h)

def dy_hat_2d(i, x, h):
    return hat(i[0], x[0], h) * grad_hat(i[1], x[1], h)


"""-------------------- QUADRATIC BASIS FUNCTIONS -----------------------"""

def quad_e(i, x, h):
    """function to generate "even" components of the quadratic basis

    Args:
        i (_float_): position of peak
        x (_float_): variable
        h (_type_): width

    Returns:
        float:
    """
    return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i+h - epsilon), x>=(i+h - epsilon)],
                           [0         , lambda x: 1 - ((x-i)/h)**2      , 0                 ])
    
def quad_o(i, x, h):
    """function to generate "odd" components of quadratic basis

    Args:
        i (_float_): position of peak
        x (_float_): variable
        h (_type_): width

    Returns:
        float:
    """
    return np.piecewise(x, [x <= (i-(2*h)), (x > (i-(2*h))) & (x <= i)                       , (x > i) & (x < (i+(2*h)))                        , x>=(i+(2*h))],
                           [0             , lambda x: (1/(2*(h**2)))*(x-(i-(2*h)))*(x-(i-h)) , lambda x: (1/(2*(h**2)))*(x-(i+(2*h)))*(x-(i+h)) , 0           ])
    
def quad(i, x, h):
    """function for vertices of quadratic basis elements

    Args:
        i (_float_): position of peak
        x (_float_): variable
        h (_type_): width

    Returns:
        float:
    """
    
    if (round(i/h) % 2) == 1:
        return quad_e(i, x, h)
    else:
        return quad_o(i, x, h)
    
def quad_2d(i, x, h):
    """2 dimensional quadratic basis function

    Args:
        i (_float_): position of peak
        x (_float_): variable
        h (_type_): width

    Returns:
        float:
    """
    return quad(i[0], x[0], h) * quad(i[1], x[1], h)

        
def psi(i, x, h, boundary=[0, 1]):
    """Piecewise Quadratic basis fuction with peak at i and element of size 2*h

    Args:
        i (float): peak of function
        x (float): variable
        h (float): half the with of element
        boundary (list, optional): boundary of domain. Defaults to [0, 1].

    Returns:
        float:
    """
    if (i == np.array(boundary)).any():
        return 0*x
    else:    
        return quad(i, x, h)
    
def grad_psi(i, x, h, boundary=[0, 1]):
    """Piecewise Quadratic basis fuction with peak at i and element of size 2*h

    Args:
        i (float): peak of function
        x (float): variable
        h (float): half the with of element
        boundary (list, optional): boundary of domain. Defaults to [0, 1].

    Returns:
        float:
    """
    if (i == np.array(boundary)).any():
        return 0*x
    else:    
        return np.gradient(quad(i, x, h), h)

    
def psi_2d(i, x, h):
    """Piecewise Quadratic 2 dimensional basis fuction with peak at i and element of size (2*h)**2

    Args:
        i (float): peak of function
        x (float): variable
        h (float): half the with of element

    Returns:
        float:
    """
    return psi(i[0], x[0], h) * psi(i[1], x[1], h)

def dx_psi_2d(i, x, h):
    return grad_psi(i[0], x[0], h) * psi(i[1], x[1], h)

def dy_psi_2d(i, x, h):
    return psi(i[0], x[0], h) * grad_psi(i[1], x[1], h)


"""-------------------- EXTRA FUNCTIONS -----------------------"""


def conv_sol(solution, x, basis_func, vertices, h):
    """Converts a solution vector (an array) into an interpolated function on x with corresponing
    basis functions (basis_func).

    Args:
        solution (numpy array): solution vector 'u' to matrix equation Au=F
        x (numpy array): domain over which we want this new function over
        basis_func (function): one of the basis functions. The function is a variable here so do not 
        put in any variables into it.
        vertices (list): list of vertices which correspont to each entry of the solution vector 'u'
        h (float): width of element. This is the same h as you would have used in defining your mesh

    Returns:
        numpy array:
    """
    
    u = 0    
    for i, vert in enumerate(vertices):
        u = u + (solution[i] * basis_func(vert, x, h))
    return u
    
    
"""-------------------- Code that works but not used in any scripts -----------------------"""

# n-dim hat basis function
def phi_Ndim(j, x, h):
    dim1_phis = []
    dim = len(j) # = len(j)
    for i in range(dim):
        dim1_phis.append(phi(j[i], x[i], h))
    
    phis_num = len(dim1_phis)
    for i in range(phis_num):
        dim1_phis[-(i+1)] = np.multiply(dim1_phis[-(i+1)], dim1_phis[-i])
    
    return dim1_phis[0]


def grad_phi_2d(i, x, h):
    return np.array([grad_phi(i[0], x[0], h) * phi(i[1], x[1], h), phi(i[0], x[0], h) * grad_phi(i[1], x[1], h)])

def grad_phi_Ndim(j, *x, h):
    dim = len(j,)
    solution = []
    
    for a in range(dim):
        grad = grad_phi(j[a], x[0][a], h)
        popped_j = list(j)
        popped_j.pop(a)
        
        for b in range(len(popped_j)):
            grad *= phi(popped_j[b], x[0][b], h)
        
        solution.append(grad)
    
    return np.array(solution)


def k_delta(i, j):
    """Kronecker delta function. When i = j return 1 else return 0.

    Args:
        i (float):
        x (float):

    Returns:
        int: 0 or 1 
    """
    
    if i == j:
        return 1
    else: 
        return 0