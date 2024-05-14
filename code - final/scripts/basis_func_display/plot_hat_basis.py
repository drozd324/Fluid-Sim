"""
Plotting of hat basis function and its derivative 
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf


N = 200
x = np.linspace(0, 1, N)
h = .3
        
plt.plot(x, bf.hat(.5, x, h), label=r"$\phi_{0.5}(x)$")
plt.plot(x, bf.grad_hat(.5, x, h), label=r"$\phi_{0.5}^\prime(x)$", linestyle="--")
plt.legend()
plt.title("the hat function and its derivative")
plt.xlabel("x")
#plt.savefig("linear_basis")
plt.show()