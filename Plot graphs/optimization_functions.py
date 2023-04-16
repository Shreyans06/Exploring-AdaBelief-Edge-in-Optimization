from autograd import grad
import autograd.numpy as np


class Beale:
    def __init__(self):
        self.x_min, self.x_max = -1.5, 5.0
        self.y_min, self.y_max = -3.0, 1.5
        self.x_start, self.y_start = 2.4, -2.8
        self.x_global_min, self.y_global_min, self.z_global_min = 3.0, 0.5, 0.0
        self._calculate_derivatives()

    def _calculate_derivatives(self):
        self.df_dx = grad(self.eval, 0)
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.log(1 + (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2) / 10
        return z


class modulus:
    def __init__(self):
        self.x_min, self.x_max = -3.0, 3.0
        self.y_min, self.y_max = -3.0, 3.0
        self.x_start, self.y_start = -2.5, 0.0
        self.x_global_min, self.y_global_min, self.z_global_min = 0.0, 0.0, 0.0
        self._calculate_derivatives()

    def _calculate_derivatives(self):
        self.df_dx = grad(self.eval, 0)
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.abs(x) + np.abs(y)
        return z

class modulus1:
    def __init__(self):
        self.x_min, self.x_max = -3.0, 3.0
        self.y_min, self.y_max = -3.0, 3.0
        self.x_start, self.y_start = 2.5, 0.0
        self.x_global_min, self.y_global_min, self.z_global_min = 0.0, 0.0, 0.0
        self._calculate_derivatives()

    def _calculate_derivatives(self):
        self.df_dx = grad(self.eval, 0)
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        list1 = [np.abs(x + y), np.abs(x - y)/10]
        z = np.sum(list1)
        return z
