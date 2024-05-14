import numpy as np
#import basis_functions as bf

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

"""-------------------------------------- APPROXIMATE VECTOR PRODUCTS -------------------------------------------------"""
# this section contains funcions which will predominatnly use the numpy gradient function to calculate derivatives

def grad_dot_grad(func0, func1, h):
    """returns a dot product of gradients of two functions of two varibles.
    ie, grad(func0) dot grad(func1)

    Args:
        func0 (function): a numpy function of two variables
        func1 (function): a numpy function of two variables

    Returns:
        numpy array:
    """
    
    dy_f0, dx_f0 = np.gradient(func0, h)
    dy_f1, dx_f1 = np.gradient(func1, h)
    
    return np.multiply(dx_f0, dx_f1) + np.multiply(dy_f0, dy_f1)

def func_mul_pdfunc(func0, func1, h, axis):
    """returns a product of a function and a partial derivative of another function.
    ie, func0 * partial_derivative(func1)

    Args:
        func0 (function): a numpy function of two variables
        func1 (function): a numpy function of two variables
        axis (0 or 1): pick 0 for partial derivative with respect to x or 1 for partial derivative with respect to y

    Returns:
        numpy array:
    """
    pd_f1 = np.gradient(func1, h, axis=axis)
    
    return np.multiply(pd_f1, func0)    


"""----------------------------------------- PRECISE VECTOR PRODUCTS ----------------------------------------------"""
# this section contains precise evaluations of important vector products using pen and paper maths as much as possible

def grad_dot_grad_phi2d(vert0, vert1, X, h):
    """returns a dot product of gradients of two bf.phi_2d functions.
    mathematically --> grad(phi2d) dot grad(phi2d)

    Args:
        vert0 (float): peak of associated basis function
        vert1 (float): peak of associated basis function
        X (numpy array): array to evaluate functions on
        h (float): width between nodes

    Returns:
        numpy array:
    """
    
    x, y = X
    i, j = vert0
    a, b = vert1

    dx_0 = bf.grad_phi(i, x, h) * bf.phi(j, y, h)
    dy_0 = bf.phi(i, x, h)      * bf.grad_phi(j, y, h)
    
    dx_1 = bf.grad_phi(a, x, h) * bf.phi(b, y, h)
    dy_1 = bf.phi(a, x, h)      * bf.grad_phi(b, y, h)
    
    return dx_0*dx_1 + dy_0*dy_1

def grad_dot_grad_psi2d(vert0, vert1, X, h):
    """returns a dot product of gradients of two bf.psi_2d functions.
    mathematically --> grad(psi2d) dot grad(psi2d)

    Args:
        vert0 (float): peak of associated basis function
        vert1 (float): peak of associated basis function
        X (numpy array): array to evaluate functions on
        h (float): width between nodes

    Returns:
        numpy array:
    """
    
    x, y = X
    i, j = vert0
    a, b = vert1

    dx_0 = bf.grad_psi(i, x, h) * bf.psi(j, y, h)
    dy_0 = bf.psi(i, x, h)      * bf.grad_psi(j, y, h)
    
    dx_1 = bf.grad_psi(a, x, h) * bf.psi(b, y, h)
    dy_1 = bf.psi(a, x, h)      * bf.grad_psi(b, y, h)
    
    return dx_0*dx_1 + dy_0*dy_1

def grad_dot_grad_hat2d(vert0, vert1, X, h):
    """returns a dot product of gradients of two bf.hat_2d functions.
    mathematically --> grad(psi2d) dot grad(psi2d)

     Args:
        vert0 (float): peak of associated basis function
        vert1 (float): peak of associated basis function
        X (numpy array): array to evaluate functions on
        h (float): width between nodes

    Returns:
        numpy array:
    """
    
    x, y = X
    i, j = vert0
    a, b = vert1

    dx_0 = bf.grad_hat(i, x, h) * bf.hat(j, y, h)
    dy_0 = bf.hat(i, x, h)      * bf.grad_hat(j, y, h)
    
    dx_1 = bf.grad_hat(a, x, h) * bf.hat(b, y, h)
    dy_1 = bf.hat(a, x, h)      * bf.grad_hat(b, y, h)
        
    return (dx_0*dx_1) + (dy_0*dy_1)

def gdg_phi_and_hat_2d(vert0, vert1, X, h, hat_bdry=[]):
    """returns a dot product of gradients of two bf.hat_2d functions.
    mathematically --> grad(phi2d) dot grad(psi2d) 

     Args:
        vert0 (float): peak of associated basis function
        vert1 (float): peak of associated basis function
        X (numpy array): array to evaluate functions on
        h (float): width between nodes
        hat_bdry (list): location of boundary for hat functions

    Returns:
        numpy array:
    """
    
    x, y = X
    i, j = vert0
    a, b = vert1

    dx_0 = bf.grad_phi(i, x, h) * bf.phi(j, y, h, hat_bdry)
    dy_0 = bf.phi(i, x, h)      * bf.grad_phi(j, y, h, hat_bdry)
    
    dx_1 = bf.grad_phi(a, x, h) * bf.phi(b, y, h, hat_bdry)
    dy_1 = bf.phi(a, x, h)      * bf.grad_phi(b, y, h, hat_bdry)
        
    return (dx_0*dx_1) + (dy_0*dy_1)