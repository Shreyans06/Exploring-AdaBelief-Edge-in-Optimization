import numpy as np
from autograd import grad


def run_optimizer(opt, cost_f, iterations):
    xs, ys = [cost_f.x_start], [cost_f.y_start]
    for epochs in range(iterations):
        x, y = opt.step()
        xs.append(x)
        ys.append(y)

    print("------------------------")
    print(cost_f.__class__.__name__)
    print(opt.__class__.__name__)
    print('X final value: ', xs[-1])
    print('Y final value: ', ys[-1])
    print("-------------------------")
    return xs, ys


class Optimizer:
    def __init__(self, cost_f, lr, x, y, **kwargs):
        self.lr = lr
        self.cost_f = cost_f
        if x == None or y == None:
            self.x = self.cost_f.x_start
            self.y = self.cost_f.y_start
        else:
            self.x = x
            self.y = y

        self.__dict__.update(kwargs)

    def step(self):
        raise NotImplementedError()


class SGD_momentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y)
        self.vx = 0.0
        self.vy = 0.0
        self.beta = beta

    def step(self):
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.vy = self.beta * self.vy + (1 - self.beta) * dy
        self.vx = self.beta * self.vx + (1 - self.beta) * dx

        # self.vx = self.beta * self.vx + self.lr * dx
        # self.vy = self.beta * self.vy + self.lr * dy

        self.x += - self.lr * self.vx
        self.y += - self.lr * self.vy

        return [self.x, self.y]


class Adam(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0.0, 0.0, 0.0, 0.0, 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def step(self):
        self.t += 1
        epsilon = 1e-8

        # derivative
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

        self.x = self.x - (self.lr * m_x_hat) / (np.sqrt(v_x_hat) + epsilon)
        self.y = self.y - (self.lr * m_y_hat) / (np.sqrt(v_y_hat) + epsilon)

        return [self.x, self.y]


class AdaBelief(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y)
        self.m_x, self.m_y, self.s_x, self.s_y, self.t = 0.0, 0.0, 0.0, 0.0, 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def step(self):
        self.t += 1
        epsilon = 1e-8

        # derivative
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy


        self.s_x = self.beta_2 * self.s_x + (1 - self.beta_2) * ((dx - self.m_x) ** 2)+ epsilon
        self.s_y = self.beta_2 * self.s_y + (1 - self.beta_2) * ((dy - self.m_y) ** 2)+ epsilon


        m_x_hat = self.m_x / (1 - (self.beta_1 ** self.t))
        m_y_hat = self.m_y / (1 - (self.beta_1 ** self.t))
        s_x_hat = self.s_x / (1 - (self.beta_2 ** self.t))
        s_y_hat = self.s_y / (1 - (self.beta_2 ** self.t))

        self.x = self.x - ((self.lr * m_x_hat) / (np.sqrt(s_x_hat) + epsilon))
        self.y = self.y - ((self.lr * m_y_hat) / (np.sqrt(s_y_hat) + epsilon))

        return [self.x, self.y]
