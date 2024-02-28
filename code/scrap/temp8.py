import numpy as np

def distFunc(x,y):
    f = np.exp(x*y)
    return f

# Values in x to evaluate the integral.
x = np.linspace(.1, 10, 100)
y = np.linspace(.1, 10, 100)
X, Y = np.meshgrid(x, y)

list1=distFunc(X, Y)
print(list1.shape)
int_exp2d = np.trapz(np.trapz(list1, y, axis=0), x, axis=0)

print(int_exp2d)