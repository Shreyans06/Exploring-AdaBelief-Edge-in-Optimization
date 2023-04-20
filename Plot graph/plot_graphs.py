from optimization_functions import Modulus, Beale, L1Loss, L2Loss, ModulusBeta
from plot_functions import plot_optimization_function, plot_path, plot_contours
from optimizer_function import run_optimizer, SGD_momentum, Adam, AdaBelief
import pickle
import os
import matplotlib.pyplot as plt

learning_rate = 0.001

# to plot contour plots and compare convergence rate of the algorithms
def contour_comparison(cost_fn,beta_1=0.9, beta_2=0.999,iterations = 2000):
    plot_optimization_function(cost_fn,name = cost_fn.__class__.__name__)

    plot_contours(cost_fn)

    opt = SGD_momentum(cost_f=cost_fn, lr=learning_rate, beta=beta_1)
    errors_momentum, distance_momentum, xs_momentum, ys_momentum = run_optimizer(opt=opt, cost_f=cost_fn,
                                                                                 iterations=iterations)

    opt = Adam(cost_f=cost_fn, lr=learning_rate,beta_1 = beta_1,beta_2 = beta_2)
    errors_adam, distance_adam, xs_adam, ys_adam = run_optimizer(opt=opt, cost_f=cost_fn, iterations=iterations)

    opt = AdaBelief(cost_f=cost_fn, lr=learning_rate,beta_1 = beta_1 , beta_2 = beta_2)
    errors_adabelief, distance_adabelief, xs_adabelief, ys_adabelief = run_optimizer(opt=opt, cost_f=cost_fn,
                                                                                     iterations=iterations)

    path_dict = {"SGD + Momentum": (xs_momentum, ys_momentum, "green"), "Adam": (xs_adam, ys_adam, "magenta"),
                 "AdaBelief": (xs_adabelief, ys_adabelief, "red") }

    pickle.dump(path_dict, open(f"./Plot values/{cost_fn.__class__.__name__}.p","wb"))

    plot_path(path_dict, cost_fn, frames=iterations, file_name=cost_fn.__class__.__name__)




# cost_f = Beale()
# contour_comparison(cost_f)
#
# cost_f = Modulus()
# contour_comparison(cost_f)
#
# cost_f = L1Loss()
# contour_comparison(cost_f)
#
# cost_f = L2Loss()
# contour_comparison(cost_f)
#
# cost_f = ModulusBeta()
# contour_comparison(cost_f,beta_1=0.3,beta_2=0.3,iterations=100)


# to plot model performance like Train accuracy vs Training epoch
def model_performance(dataset='CIFAR-10', metric='test_acc'):
    file_names = [name for name in os.listdir('./Plot_curves')]
    files = {}

    for names in file_names:
        if names.split("_")[0] != dataset:
            return 0
        files[names.split(".")[0]] = pickle.load(open('./Plot_curves/' + names, "rb"))
        print(files)
        plt.plot([j for j in range(len(files[names.split(".")[0]][metric]))], files[names.split(".")[0]][metric],
                 linewidth=2, label=names.split(".")[0].split("_")[-1])

    plt.grid()
    plt.ylim(84, 96)
    plt.legend(fontsize=14)
    plt.title(f'{metric} ~ Training epoch')
    plt.legend(loc="lower right")
    plt.xlabel('Training Epoch')
    plt.ylabel(f'{metric}')
    plt.show()

model_performance()
