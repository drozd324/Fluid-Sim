import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True

N = 30
fps = 100
frn = 160

x = np.linspace(-4, 4, N + 1)
x, y = np.meshgrid(x, x)
zarray = np.zeros((N + 1, N + 1, frn))

def f(x, y, t):
   return (1-t)*np.exp(-(x**2 + y**2))

for i in range(frn):
   #print(f(x, y, i/frn).shape)
   zarray[:, :, i] = f(x, y, i/frn)
   
def change_plot(frame_number, zarray, plot):
   plot[0].remove()
   plot[0] = ax.plot_surface(x, y, zarray[:, :, frame_number], cmap="afmhot_r", vmin = 0, vmax = 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#print(zarray[:, :, 0].shape)
#print(zarray[:, :, 0])

plot = [ax.plot_surface(x, y, zarray[:, :, 0], vmin = 0, vmax = 1)]
ax.set_zlim(0, 1)
ax.axis('off')
ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(zarray, plot), interval=1000 / fps)

fn = 'plot_surface_animation_funcanimation'
ani.save(fn+'.gif',writer='PillowWriter',fps=fps, dpi=200)

plt.show()