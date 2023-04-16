import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, correct_bias=True):

        if lr < 0:
            raise ValueError("Invalid learning rate")
        if not 0 <= betas[0] <= 1:
            raise ValueError("Invalid beta 0")
        if not 0 <= betas[1] <= 1:
            raise ValueError("Invalid beta 1")
        if not 0 <= eps:
            raise ValueError("Invalid epsilon")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)

        super(AdaBelief, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Grads are sparse')

                state = self.state[param]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta_1, beta_2 = group['betas']
                state['step'] += 1

                m_t = torch.add(torch.mul(beta_1, exp_avg), torch.mul(1.0 - beta_1, grad))
                # s_t = torch.add(torch.mul(beta_2, exp_avg_sq) , torch.mul(1.0-beta_2,torch.square(grad)) )
                s_t = torch.add(torch.add(torch.mul(beta_2, exp_avg_sq),
                                          torch.mul(1.0 - beta_2, torch.square(torch.sub(grad, m_t)))), group['eps'])

                if group['correct_bias']:
                    m_t = m_t.divide(1.0 - beta_1 ** state['step'])
                    s_t = s_t.divide(1.0 - beta_2 ** state['step'])

                denom = torch.add(torch.sqrt(s_t), group['eps'])
                param.data.addcdiv_(m_t, denom, value=-group['lr'])