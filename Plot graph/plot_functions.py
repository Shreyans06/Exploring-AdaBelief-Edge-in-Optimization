import matplotlib
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

    min_val, max_val = 0.2, 0.6
    n = 20
    orig_cmap = plt.cm.Purples
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    #
    # orig_cmap = plt.cm.aqua
    # colors1 = plt.aqua(np.linspace(min_val, max_val, 10))
    # print(colors1)
    # print(colors)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors = colors)

    # gradient = np.linspace(0, 1, 256)
    # gradient = np.vstack((gradient, gradient))
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 2))
    # ax1.imshow(gradient, cmap=orig_cmap, aspect='auto')
    # ax1.set_title('original')
    # ax2.imshow(gradient, cmap=cmap, aspect='auto')
    # ax2.set_title('custom')
    # plt.tight_layout()

    ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(z_array.min(), z_array.max(), 20))
    ax.text(optim_fn.x_global_min, optim_fn.y_global_min, "X",horizontalalignment='center',verticalalignment='center',color="black", size=15)

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

    animation = FuncAnimation(fig, animate, interval=0.5, blit=True, repeat=True, frames=frames)
    plt.title(f"Loss function : {file_name}")
    animation.save(f"./Convergence/{file_name}.gif", dpi = 400, writer=PillowWriter(fps=100))
    fig.savefig(f"./Trajectories/{file_name}.png", dpi = 400)
    plt.close(fig)


