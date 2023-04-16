from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from autograd import grad


def plot_optimization_function(optim_fn, steps=0.01, figsize=[10, 10]):
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

    plt.show()


def plot_contours(optim_fn, steps=0.01, figsize=[10, 10]):
    x_grid = np.arange(optim_fn.x_min, optim_fn.x_max, steps)
    y_grid = np.arange(optim_fn.y_min, optim_fn.y_max, steps)

    X, Y = np.meshgrid(x_grid, y_grid)

    z_array = np.array([optim_fn.eval(x_grid, y_grid) for x_grid, y_grid in zip(np.ravel(X), np.ravel(Y))])
    Z = z_array.reshape(X.shape)
    fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot()

    plt.contourf(X, Y, Z, cmap=plt.cm.afmhot, levels=np.linspace(z_array.min(), z_array.max(), 1000))
    plt.text(optim_fn.x_global_min, optim_fn.y_global_min, "X", color="purple", size=20)

    #     plt.show()

    return fig, plt


def plot_path(path_dict, optim_fn, figsize=[10, 10], frames=7, file_name=''):
    fig, ax = plot_contours(optim_fn)

    global dots;
    dots = []

    def update(frame_number):
        global dots
        ax = fig.gca()
        for sc in dots: sc.remove()
        dots = []

        for name, (x, y, c) in path_dict.items():
            ax.plot(x[:frame_number], y[:frame_number], color=c, zorder=1, linewidth=2)
            k = ax.scatter(x[frame_number], y[frame_number], color=c, zorder=1, s=50)
            dots.append(k)

    color_patch = []
    for algo, (x, y, c) in path_dict.items():
        color_patch.append(mlines.Line2D([], [], color=c, label=algo))
    ax.legend(handles=color_patch)

    animation = FuncAnimation(fig, update, interval=1, frames=frames)
    animation.save(f"./{file_name}.gif", dpi=80, writer="imagemagick",fps=24)
