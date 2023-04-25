from optimization_functions import Modulus, Beale, L1Loss, L2Loss, ModulusBeta
from plot_functions import plot_optimization_function, plot_path, plot_contours
from optimizer_function import run_optimizer, SGD_momentum, Adam, AdaBelief
import pickle
import os
import matplotlib.pyplot as plt

learning_rate = 0.0005

# to plot contour plots and compare convergence rate of the algorithms
def contour_comparison(cost_fn,beta_1=0.9, beta_2=0.999,iterations = 2000):
    plot_optimization_function(cost_fn,name = cost_fn.__class__.__name__)

    plot_contours(cost_fn)

    sgd_opt = SGD_momentum(cost_f=cost_fn, lr=learning_rate, beta=beta_1)
    xs_momentum, ys_momentum = run_optimizer(opt=sgd_opt, cost_f=cost_fn,iterations=iterations)

    adam_opt = Adam(cost_f=cost_fn, lr=learning_rate,beta_1 = beta_1,beta_2 = beta_2)
    xs_adam, ys_adam = run_optimizer(opt=adam_opt, cost_f=cost_fn, iterations=iterations)

    adabelief_opt = AdaBelief(cost_f=cost_fn, lr=learning_rate,beta_1 = beta_1 , beta_2 = beta_2)
    xs_adabelief, ys_adabelief = run_optimizer(opt=adabelief_opt, cost_f=cost_fn,iterations=iterations)

    path_dict = {"SGD + Momentum": (xs_momentum, ys_momentum, "blue"), "Adam": (xs_adam, ys_adam, "green"),
                 "AdaBelief": (xs_adabelief, ys_adabelief, "red") }

    pickle.dump(path_dict, open(f"./Plot values/{cost_fn.__class__.__name__}.p","wb"))

    plot_path(path_dict, cost_fn, frames=iterations, file_name=cost_fn.__class__.__name__)


# cost_f = Beale()
# contour_comparison(cost_f)

cost_f = Modulus()
contour_comparison(cost_f,iterations=500)

# cost_f = L1Loss()
# contour_comparison(cost_f,iterations=2500)
#
# cost_f = L2Loss()
# contour_comparison(cost_f)
#
# cost_f = ModulusBeta()
# contour_comparison(cost_f,beta_1=0.3,beta_2=0.3,iterations=100)


# to plot model performance like Train accuracy vs Training epoch
def model_performance(dataset='CIFAR-10', architecture = 'VGG', metric='test_acc'):
    file_names = [name for name in os.listdir('./Plot_curves')]
    files = {}
    if metric == 'test_acc':
        type = 'Test Accuracy'
    elif metric == 'train_acc':
        type = 'Train Accuracy'

    for names in file_names:
        if names.split("_")[0] == dataset and names.split("_")[1] == architecture:
            files[names.split(".")[0]] = pickle.load(open('./Plot_curves/' + names, "rb"))
            style ='--'
            if names.split(".")[0].split("_")[-1] == 'AdaBelief':
                style= '-'
            plt.plot([j for j in range(len(files[names.split(".")[0]][metric]))], files[names.split(".")[0]][metric],
                         linewidth=2, label=names.split(".")[0].split("_")[-1], linestyle=style)

    plt.grid()
    y_lim =(82,98)
    if metric == 'test_acc' and  dataset=='CIFAR-10' and architecture == 'VGG' :
        y_lim = (84,92)
    elif metric == 'test_acc' and  dataset=='CIFAR-10' and architecture == 'ResNet' :
        y_lim = (88,96)
    plt.ylim(y_lim)
    plt.legend(fontsize=14)
    plt.legend(loc="upper left")
    plt.xlabel('Training Epoch')
    plt.ylabel(f'{type}')
    plt.title(f'{architecture} on {dataset} - {type} ~ Training epoch')
    plt.savefig(f'./Plot_curves/{architecture} on {dataset} - {type} ~ Training epoch', dpi=400)
    plt.show()

# model_performance(architecture='ResNet')
