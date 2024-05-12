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

N = 1000
x = np.linspace(0, 1, N)

H = 3
vertices = np.linspace(0, 1, H)
h = vertices[1] - vertices[0]


# these wont go into the for loop below because formatting a latex string is a pain in the ass
plt.plot(x, bf.quad(0, x, h), label=r"$\psi_{0}(x)$", linestyle="--")
plt.plot(x, 3*bf.quad(.5, x, h), label=r"$\psi_{0.5}(x)$", linestyle="--")
plt.plot(x, 2*bf.quad(1, x, h), label=r"$\psi_{1}(x)$", linestyle="--")

u = 0
for l, i in enumerate([0, .5, 1]):
    func = bf.quad(i, x, h)
    u = u + func

plt.plot(x, u, label="sum of bases")
plt.xlabel("x")
plt.legend()
plt.savefig("quad_basis")
plt.show()