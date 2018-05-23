"""
code based from 
http://pytorch.org/docs/master/_modules/torch/optim/sgd.html#SGD
"""
from __future__ import division
from torch.optim.optimizer import Optimizer, required


class NoisedSGD(Optimizer):

    def __init__(self, params, lr=required, dataset_size=required,momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, ):
        self.correction = dataset_size / 2 ## for the 2 sources of noise in sgld to be balanced
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NoisedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NoisedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                noise = d_p.new(d_p.size()).normal_(0, (0.5 * group['lr']/self.correction)**.5)
                p.data.add_(-group['lr']/2 , d_p)
                p.data.add_(-1, noise)

        return loss
