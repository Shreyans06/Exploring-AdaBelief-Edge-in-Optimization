import numpy as np
from autograd import grad

def run_optimizer(opt, cost_f, iterations, *args, **kwargs):
    errors = [cost_f.eval(cost_f.x_start, cost_f.y_start)]
    xs, ys = [cost_f.x_start], [cost_f.y_start]
    for epochs in range(iterations):
        x, y = opt.step(*args, **kwargs)
        xs.append(x)
        ys.append(y)
        errors.append(cost_f.eval(x, y))
    distance = np.sqrt((np.array(xs) - cost_f.x_global_min) ** 2 + (np.array(ys) - cost_f.y_global_min) ** 2)
    return errors, distance, xs, ys


class Optimizer:
    def __init__(self, cost_f, lr, x, y, **kwargs):
        self.lr = lr
        self.cost_f = cost_f
        if x == None or y == None:
            self.x = self.cost_f.x_start
            self.y = self.cost_f.y_start
        else:
            self.x = float(x)
            self.y = float(y)

        self.__dict__.update(kwargs)

    def step(self, lr):
        raise NotImplementedError()


class SGD_momentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y, beta=beta)
        self.vx = 0.0
        self.vy = 0.0

    def step(self, lr=None, beta=None):
        if type(lr) == type(None):
            lr = self.lr
        if type(beta) == type(None):
            beta = self.beta
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.vx = beta * self.vx + (1-beta) * dx
        self.vy = beta * self.vy + (1-beta) * dy
        self.x += - lr * self.vx
        self.y += - lr * self.vy

        return [self.x, self.y]


class Adam(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0.0, 0.0, 0.0, 0.0, 0.0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy
        self.v_x = self.beta_2 * self.v_x + (1 - self.beta_2) * (dx ** 2)
        self.v_y = self.beta_2 * self.v_y + (1 - self.beta_2) * (dy ** 2)

        m_x_hat = self.m_x / (1 - self.beta_1 ** self.t)
        m_y_hat = self.m_y / (1 - self.beta_1 ** self.t)
        v_x_hat = self.v_x / (1 - self.beta_2 ** self.t)
        v_y_hat = self.v_y / (1 - self.beta_2 ** self.t)

        self.x = self.x - (lr * m_x_hat) / (np.sqrt(v_x_hat) + epsilon)
        self.y = self.y - (lr * m_y_hat) / (np.sqrt(v_y_hat) + epsilon)
        return [self.x, self.y]


class AdaBelief(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.s_x, self.s_y, self.t = 0.0, 0.0, 0.0, 0.0, 0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr

        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy
        self.s_x = self.beta_2 * self.s_x + (1 - self.beta_2) * ((dx - self.m_x) ** 2) + epsilon
        self.s_y = self.beta_2 * self.s_y + (1 - self.beta_2) * ((dy - self.m_y) ** 2) + epsilon

        m_x_hat = self.m_x / (1 - self.beta_1 ** self.t)
        m_y_hat = self.m_y / (1 - self.beta_1 ** self.t)
        s_x_hat = self.s_x / (1 - self.beta_2 ** self.t)
        s_y_hat = self.s_y / (1 - self.beta_2 ** self.t)

        self.x = self.x - ((lr * m_x_hat) / (np.sqrt(s_x_hat) + epsilon))
        self.y = self.y - ((lr * m_y_hat) / (np.sqrt(s_y_hat) + epsilon))
        return [self.x, self.y]