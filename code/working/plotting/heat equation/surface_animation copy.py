import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

#plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True

u_sols = torch.load("heat_equation.pt")

u_0 = u_sols[0]

u_0[:, -1] = 0
u_0[:, 0] = 0
u_0[0, :] = 0
u_0[-1, :] = 0

u_sols[0] = u_0

N = 20
len_u_sols = len(u_sols)
print(len(u_sols))
frn = int(len_u_sols/2)
fps = int(frn/6)

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

fn = 'plot'
ani.save(fn+'.gif',writer='PillowWriter',fps=fps, dpi=300)

plt.show()