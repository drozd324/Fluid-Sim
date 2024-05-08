import numpy as np

"""import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf"""

def l_squared_norm(f, X0, x0):
    return np.sqrt( np.trapz(np.trapz( f(X0)**2 , x0[0], axis=0), x0[1], axis=0) )