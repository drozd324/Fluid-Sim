import numpy as np
import matplotlib.pyplot as plt

#import basis_functions as bf
    
# defining mesh
H = 1000
h = .1
x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y)
num_vertices = H**2


def quad_e(x, i, h):
    return np.piecewise(x, [x <= (i-h), (x > (i-h))&(x <= i+h)        , x>=(i+h)],
                            [0         , lambda x: 1 - ((x-i)/h)**2 , 0       ])

def quad_o(x, i, h):
    return np.piecewise(x, [x <= (i-(2*h)), (x > (i-(2*h))) & (x <= i)                       , (x > i) & (x < (i+(2*h)))                        , x>=(i+(2*h))],
                           [0             , lambda x: (1/(2*(h**2)))*(x-(i-(2*h)))*(x-(i-h)) , lambda x: (1/(2*(h**2)))*(x-(i+(2*h)))*(x-(i+h)) , 0       ])

def quad(x, i, h):
    if (round(i/h) % 2) == 1:
        return quad_e(x, i, h)
    else:
        return quad_o(x, i, h)

plat = 0
for s, i in zip([0, 1, 2, 3, 4, 3, 2, 1, 1 ,2 ,0], [j/10 for j in range(11)]):
    plat += s*quad(x, i, h)
plt.plot(x, plat)

plt.axhline(y=0, color="black")
plt.axvline(x=0, color="black")

#plt.vlines()
#plt.xlim(0, 1)
plt.show()