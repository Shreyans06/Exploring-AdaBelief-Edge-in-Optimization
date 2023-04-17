from autograd import grad
import autograd.numpy as np


class Beale:
    def __init__(self):
        self.x_min, self.x_max = -4.0, 4.0
        self.y_min, self.y_max = -4.0, 4.0
        self.x_start, self.y_start = -2.5, -2.5
        self.x_global_min, self.y_global_min, self.z_global_min = 3.0, 0.5, 0.0
        self._calculate_derivatives()

    def _calculate_derivatives(self):
        self.df_dx = grad(self.eval, 0)
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.log(1 + (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2) / 10
        return z


class Modulus:
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
        f = [np.abs(x), np.abs(y)]
        z = np.sum(f)
        return z


class L1Loss:
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
        f = [np.abs(x + y), np.abs(x - y) / 10]
        z = np.sum(f)
        return z

class L2Loss:
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
        f = [(x + y) ** 2, (x - y) ** 2 / 10]
        z = np.sum(f)
        return z

class ModulusBeta:
    def __init__(self):
        self.x_min, self.x_max = -3.0, 3.0
        self.y_min, self.y_max = -3.0, 3.0
        self.x_start, self.y_start = 0.5, -1.5
        self.x_global_min, self.y_global_min, self.z_global_min = 0.0, 0.0, 0.0
        self._calculate_derivatives()

    def _calculate_derivatives(self):
        self.df_dx = grad(self.eval, 0)
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        f = [np.abs(x) / 10, np.abs(y)]
        z = np.sum(f)
        return z
