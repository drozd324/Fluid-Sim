import numpy as np

"""-------------------- Good and documented functions -----------------------"""

def phi(i, x, h):
    """hat function with peak at i.

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i)        , (x > (i))&(x < (i+h))          , x>=(i+h)],
                               [0         , lambda x: (x/h) + (1-(i/h)) , lambda x: (-x/h) + (1+(i/h))   , 0       ])

def phi_2d(i, x, h):
    """2 dimensional hat function with peak at i.

    Args:
        i (tuple): position of peak
        x (tuple): variable

    Returns:
        float: 
    """
    return phi(i[0], x[0], h) * phi(i[1], x[1], h)

def grad_phi(i, x, h):
    """derivative of hat function with peak at i 

    Args:
        i (float): position of peak
        x (float): variable

    Returns:
        float: 
    """
    if i==0 or i==1:# 0,1 are the boundary conditions
        return 0 * x
    else:
        return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i), (x > (i))&(x < (i+h)), x>=(i+h)],
                               [0         , 1/h                 , -1/h                 , 0       ])

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


"""-------------------- Code that may work but not used in any scripts -----------------------"""

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
