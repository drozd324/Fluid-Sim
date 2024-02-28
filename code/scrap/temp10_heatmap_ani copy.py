import matplotlib.pyplot as plt
import numpy as np


for i in range(10):
    mat = np.random.rand(10,10)
    
    plt.imshow(mat)    
    plt.pause(0.1)
    