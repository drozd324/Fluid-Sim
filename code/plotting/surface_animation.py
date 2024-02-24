import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

#plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True

u_sols = torch.load("time_dep_poisson.pt")

print(u_sols[0].shape)

N = 20
fps = 4
frn = 10

x = np.linspace(0, 1, N)
x, y = np.meshgrid(x, x)

def change_plot(frame_number, u_sols, plot):
   plot[0].remove()
   plot[0] = ax.plot_surface(x, y, u_sols[frame_number], cmap="plasma", vmin=0, vmax=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plot = [ax.plot_surface(x, y, u_sols[0])]
ax.set_zlim(0, 1)
ax.axis('off')

ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(u_sols, plot), interval=1000 / fps)

fn = 'plot_surface_animation_funcanimation'
ani.save(fn+'.gif',writer='PillowWriter',fps=fps, dpi=300)

plt.show()