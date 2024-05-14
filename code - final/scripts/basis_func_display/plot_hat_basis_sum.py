"""
Plotting of sum of hat basis fuctions to show linear interpolation between their peaks
"""

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

H = 5
vertices = np.linspace(0, 1, H)
h = vertices[1] - vertices[0]

func1 = 1 * bf.phi(.25, x, h)
func2 = 3 * bf.phi(.5, x, h)
func3 = 2 * bf.phi(.75, x, h)
func_sum = func1 + func2 + func3

plt.plot(x, func_sum, label="sum of bases", linestyle="-")
plt.plot(x, func1, label=r"$1\phi_{0.25}(x)$", linestyle="--")
plt.plot(x, func2, label=r"$3\phi_{0.5}(x)$" , linestyle="-.")
plt.plot(x, func3, label=r"$2\phi_{0.75}(x)$", linestyle=":")
plt.xlabel("x")
plt.legend()
#plt.savefig("linear_basis_sum")
plt.show()