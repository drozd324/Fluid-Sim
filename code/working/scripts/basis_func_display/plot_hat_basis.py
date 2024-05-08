import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

"""
so what da fuck did i learn 
use h = x[1] - x[0]
linspace makes more sense
do NOT USE += ON NUMPY ARRAYS
"""

N = 200
x = np.linspace(0, 1, N)
h = .2
        
plt.plot(x, bf.hat(.5, x, h), label=r"$\phi_{0.5}(x)$")
plt.plot(x, bf.grad_hat(.5, x, h), label=r"$\phi_{0.5}^\prime(x)$", linestyle="--")
#plt.plot(x, np.gradient(bf.hat(.5, x, h), x), label="yee", linestyle="--")
plt.legend()
plt.title(r"$\phi_i(x)$ is the hat functon with peak at i")
plt.xlabel("x")
#plt.savefig("linear_basis")
plt.show()