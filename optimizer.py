import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4,
                 weight_decouple=True, fixed_decay=False, correct_bias=True, rectify=True):

        if lr < 0:
            raise ValueError("Invalid learning rate")
        if not 0 <= betas[0] <= 1:
            raise ValueError("Invalid beta 0")
        if not 0 <= betas[1] <= 1:
            raise ValueError("Invalid beta 1")
        if not 0 <= eps:
            raise ValueError("Invalid epsilon")

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias, buffer=[[None, None, None] for _ in range(10)])

        super(AdaBelief, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify

    def reset(self):
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)

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

                #                 Weight Decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        param.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        param.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(param.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta_1, beta_2 = group['betas']
                #                 state['step'] += 1

                #                 m_t = torch.add(torch.mul(beta_1, exp_avg), torch.mul(1.0 - beta_1, grad))
                #                 # s_t = torch.add(torch.mul(beta_2, exp_avg_sq) , torch.mul(1.0-beta_2,torch.square(grad)) )
                #                 s_t = torch.add(torch.add(torch.mul(beta_2, exp_avg_sq),
                #                                           torch.mul(1.0 - beta_2, torch.square(torch.sub(grad, m_t)))), group['eps'])

                #                 if group['correct_bias']:
                #                     m_t_hat = m_t.divide(1.0 - beta_1 ** state['step'])
                #                     s_t_hat = s_t.divide(1.0 - beta_2 ** state['step'])

                state['step'] += 1
                bias_correction1 = 1 - beta_1 ** state['step']
                bias_correction2 = 1 - beta_2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta_2).addcmul_(grad_residual, grad_residual, value=1 - beta_2)

                denom = (exp_avg_sq.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    param.data.addcdiv_(exp_avg, denom, value=-step_size)


                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta_2_t = beta_2 ** state['step']
                        N_sma_max = 2 / (1 - beta_2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta_2_t / (1 - beta_2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta_2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta_1 ** state['step'])
                        #                         elif self.degenerated_to_sgd:
                        #                             step_size = 1.0 / (1 - beta_1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        param.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        param.data.add_(exp_avg, alpha=-step_size * group['lr'])

#                 denom = torch.add(torch.sqrt(s_t_hat), group['eps'])
#                 param.data.addcdiv_(m_t_hat, denom, value=-group['lr'])