import numpy as np

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = np.array([1,2,3,4])

print((a == b ).all())  #False
print((a == c ).all())   # True
print((a == b ).any())   #False
print((a == c ).any())   #True
print((a > 3 ).all())    #False