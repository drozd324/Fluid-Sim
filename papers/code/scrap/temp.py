import numpy as np
import scipy
N = 5
def f(t, x):
   return np.exp(-x*t) / t**N

solution = scipy.integrate.nquad(f, [[1, np.inf], [0, np.inf]])

print(solution)