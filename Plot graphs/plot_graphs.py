
from optimization_functions import Beale,modulus,modulus1
from plot_functions import plot_optimization_function,plot_path,plot_contours
from optimizer_function import run_optimizer,SGD_momentum,Adam,AdaBelief
import pickle
import os
import matplotlib.pyplot as plt
iterations = 1500
learning_rate = 0.001

# to plot contour plots and compare convergence rate of the algorithms
def contour_comparison(cost_fn):
    plot_optimization_function(cost_fn)
    # plt.show()

    plot_contours(cost_fn)
    # plt.show()


    opt = SGD_momentum(cost_f=cost_fn, lr=learning_rate, beta=0.9)
    errors_momentum, distance_momentum,xs_momentum, ys_momentum = run_optimizer(opt=opt, cost_f=cost_fn, iterations=iterations)

    opt = Adam(cost_f=cost_fn, lr=learning_rate)
    errors_adam, distance_adam,xs_adam, ys_adam = run_optimizer(opt=opt, cost_f=cost_fn, iterations=iterations)

    opt = AdaBelief(cost_f=cost_fn, lr=learning_rate)
    errors_adabelief, distance_adabelief,xs_adabelief, ys_adabelief = run_optimizer(opt=opt, cost_f=cost_fn, iterations=iterations)


    trajectories_dict={"SGD": (xs_momentum, ys_momentum, "b") , "Adam" : (xs_adam,ys_adam,"green") , "AdaBelief" : (xs_adabelief,ys_adabelief,"pink") }
    plot_path(trajectories_dict, cost_fn, figsize=[10,10], frames=1000,file_name = cost_fn.__class__.__name__)


# cost_fn = modulus1()
# contour_comparison(cost_fn)

# to plot model performance like Train accuracy vs Training epoch
def model_performance(dataset='CIFAR-10',metric='test_acc'):
    file_names = [name for name in os.listdir('./Plot_curves')]
    files = {}

    for names in file_names:
        if names.split("_")[0] != dataset:
            return 0
        files[names.split(".")[0]] = pickle.load(open('./Plot_curves/' + names , "rb"))
        plt.plot([ j for j in range(len(files[names.split(".")[0]][metric]))] ,files[names.split(".")[0]][metric],linewidth = 2,label= names.split(".")[0].split("_")[-1])

    plt.grid()
    plt.ylim(80,101)
    plt.legend(fontsize=14)
    plt.title(f'{metric} ~ Training epoch')
    plt.legend(loc="lower right")
    plt.xlabel('Training Epoch')
    plt.ylabel(f'{metric}')
    plt.show()

model_performance()