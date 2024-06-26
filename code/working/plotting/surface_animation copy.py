import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

#plt.rcParams["figure.figsize"] = [8, 4]
#plt.rcParams["figure.autolayout"] = True

u_sols = torch.load("heat_equation_ele_2.pt")

H = 7

len_u_sols = len(u_sols)
frn = len_u_sols
fps = frn//3

x = np.linspace(0, 1, H)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y)

def change_plot(frame_number, u_sols, plot):
   plot[0].remove()  
   plot[0] = ax.plot_surface(X, Y, u_sols[frame_number], cmap="plasma", vmin=0, vmax=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot = [ax.plot_surface(X, Y, u_sols[0])]
ax.set_zlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y, t)")
ax.set_title(r"$ \frac{\partial u}{\partial t} = \nabla^2 u $ with $u(x,y,0) = sin(\pi x)sin(\pi y)$")
#ax.axis('off')

ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(u_sols, plot), interval=1000 / fps)

fn = 'plot_ele'
ani.save(fn+'.gif',writer='PillowWriter',fps=fps, dpi=300)

plt.show()