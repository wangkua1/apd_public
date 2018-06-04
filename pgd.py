'''
PGD Adversarial Attack
'''

from collections import Iterable

import torch
from torch.autograd import Variable, grad
import numpy as np

from foolbox.attacks.base import Attack, call_decorator
from foolbox.criteria import Misclassification

class PGDAttack(object):

    def __init__(self, model, epsilon, lr, steps=40, random_restart=None):
        self.model = model

        self.epsilon = epsilon
        self.steps = steps
        self.random_restart = random_restart
        self.lr = lr
    
    def attack(self, images, labels, loss_fn, *args):
        max_found = -np.inf

        images_nat = images
        if self.random_restart:
            uni_noise = self.epsilon * (2 *  torch.rand(images.size()) - 1)
            images = images + uni_noise
        
        images = Variable(images)

        # Should take n steps while clamping inside the epsilon ball each time
        for i in range(self.steps):
            preds = self.model(images)
            loss = loss_fn(preds, labels, *args)
            x_grad = grad(loss, images)

            images = images + self.lr * torch.sign(x_grad)
            images = torch.clamp(images, images_nat - self.epsilon, images_nat + self.epsilon)
        return images

class FBOXPGDAttack(Attack):
    """Like GradientAttack but with several steps for each epsilon.
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, epsilon=0.3, steps=40, lr=0.01):
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()
        perturbed = image

        for i in range(steps):
            gradient = np.sign(a.gradient(perturbed)) * (max_ - min_)
            perturbed = perturbed + gradient * lr
            perturbed = np.clip(perturbed, image - epsilon, image + epsilon)
            perturbed = np.clip(perturbed, min_, max_)
            _, is_adversarial = a.predictions(perturbed)
