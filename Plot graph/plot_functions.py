from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
import matplotlib.lines as mlines
from autograd import grad


def plot_optimization_function(optim_fn, steps=0.01, figsize=[10, 10], name=''):
    x_grid = np.arange(optim_fn.x_min, optim_fn.x_max, steps)
    y_grid = np.arange(optim_fn.y_min, optim_fn.y_max, steps)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([optim_fn.eval(x_grid, y_grid) for x_grid, y_grid in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, shade=False, rstride=5, cstride=1)
    ax.patch.set_facecolor('white')
    ax.view_init(elev=30, azim=60)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Loss function : {name}")
    plt.savefig(f"./Loss functions/{name}.png")


def plot_contours(optim_fn, steps=0.01):
    x_grid = np.arange(optim_fn.x_min, optim_fn.x_max, steps)
    y_grid = np.arange(optim_fn.y_min, optim_fn.y_max, steps)

    X, Y = np.meshgrid(x_grid, y_grid)

    z_array = np.array([optim_fn.eval(x_grid, y_grid) for x_grid, y_grid in zip(np.ravel(X), np.ravel(Y))])
    Z = z_array.reshape(X.shape)

    fig, ax = plt.subplots()

    ax.contourf(X, Y, Z, cmap="Blues_r", levels=np.linspace(z_array.min(), z_array.max(), 20))
    ax.text(optim_fn.x_global_min, optim_fn.y_global_min, "X",horizontalalignment='center',verticalalignment='center',color="orange", size=15)

    return fig, ax


def plot_path(path_dict, optim_fn, frames=7, file_name=''):
    fig, ax1 = plot_contours(optim_fn)

    ax = ax1.twinx()
    y_m = ax1.get_ylim()

    def animate(i):
        ax.clear()
        ax.set_ylim(y_m)
        plots= []
        for name, (x, y, c) in path_dict.items():
            line, = ax.plot(x[:i], y[:i], color=c, lw=1)
            point, = ax.plot(x[i], y[i], color=c, marker='.')
            plots.append(line)
            plots.append(point)

        color_patch = []
        for algo, (x, y, c) in path_dict.items():
            color_patch.append(mlines.Line2D([], [], color=c, label=algo))
        ax.legend(handles=color_patch)

        return plots

    color_patch = []
    for algo, (x, y, c) in path_dict.items():
        color_patch.append(mlines.Line2D([], [], color=c, label=algo))
    ax.legend(handles=color_patch)

    animation = FuncAnimation(fig, animate, interval=1, blit=True, repeat=True, frames=frames)
    plt.title(f"Loss function : {file_name}")
    animation.save(f"./Convergence/{file_name}.gif", dpi=300, writer=PillowWriter(fps=100))
    fig.savefig(f"./Trajectories/{file_name}.png")
    plt.close(fig)


