import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(top_dir)

from tools import basis_functions as bf

N = 1000
x = np.linspace(0, 1, N)

H = 6
vertices = np.linspace(0, 1, H)
h = vertices[1] - vertices[0]

func2 = 1 * bf.quad_o(.5, x, h)

plt.plot(x, func2, label=r"$odd$")

plt.xlabel("x")
plt.savefig("quad_basis_odd")
plt.show()