"""
Defining norms on functions to find error in calcualtions
"""

import numpy as np

def l_squ_norm(f, x0):
    """The l squared norm for functions of one variable, or here for 1 dimensional arrays.

    Args:
        f (numpy array): function given by numpy array
        x0 (numpy array): array on which the function is defined on

    Returns:
        float: 
    """
    return np.trapz( np.abs(f)**2 , x0, axis=0)

def l_squ_norm_2d(f, x0):
    """The l squared norm for functions of two variable, or here for 2 dimensional arrays.

    Args:
        f (numpy array): function given by numpy array
        x0 (numpy array): tuple of arrays on which the function is defined on

    Returns:
        float: 
    """
    
    return np.sqrt( np.trapz(np.trapz( np.abs(f)**2 , x0[0], axis=0), x0[1], axis=0) )