import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt

xlen = 10
ylen = 16
xPoints = np.arange(0,xlen+1,1)
yPoints = np.arange(0,ylen+1,1)

gridPoints = np.array([[[x,y] for y in yPoints] for x in xPoints])

a = [[i+j*(ylen+1),(i+1)+j*(ylen+1),i+(j+1)*(ylen+1)] for i in range(ylen) for j in range(xlen)]
triang = tri.Triangulation(gridPoints[:,:,0].flatten(), gridPoints[:,:,1].flatten(),a)

plt.triplot(triang)
plt.plot(gridPoints[:,:,0],gridPoints[:,:,1],'bo')
plt.title("Triangulation Visualization")
plt.show()