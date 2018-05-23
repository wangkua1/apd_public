"""gan_pytorch.py


Usage:
    gan_pytorch.py <src_dir> <f_opt_config> [--cuda]


Example:
    python gan_pytorch.py CE.fc1-100-X-nsgd-bdk-X-babymnist@2017-11-25 wgan 10000 200 babymnist
    python gan_pytorch.py CE.fc1-mnist-100-X-nsgd-bdk-X-mnist@2017-11-24 wgan 10000 200 mnist
"""

import os
import sys
import pdb
import time
import itertools
import pickle
from docopt import docopt
from collections import OrderedDict,defaultdict
import yaml

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

import tflib.plot
import tflib as lib
from tqdm import tqdm

import sklearn.metrics as sk

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# Local imports
from model.fc import *
from opt.loss import *
import gan_utils
import utils
DATA_DIR = './data'
SAVED_SAMPLE_DIR = 'saves'
GAN_EXPERIMENT_DIR = 'gan_exps'


def data_generator(src_dir, batch_size, N_TRAIN):
    f_data = [f for f in os.listdir(os.path.join(SAVED_SAMPLE_DIR, src_dir)) if f.endswith('npy')]
    all_file_data = [np.load(os.path.join(SAVED_SAMPLE_DIR, src_dir, f_datum)) for f_datum in f_data]
    data = np.concatenate(all_file_data, 0)
    data = data[:N_TRAIN]

    def get_epoch():
        # rng_state = np.random.get_state()
        np.random.shuffle(data)
        # np.random.set_state(rng_state)
        for i in range(int(len(data) / batch_size)):
            yield data[i*batch_size:(i+1)*batch_size]

    return get_epoch





def ReLULayer(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.kaiming_uniform(layer.weight.data)  # He initialization
    return nn.Sequential(layer, nn.ReLU())


class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.output_size = output_size

        self.model = nn.Sequential(
                        ReLULayer(noise_size, hidden_size),
                        ReLULayer(hidden_size, hidden_size),
                        ReLULayer(hidden_size, hidden_size),
                        nn.Linear(hidden_size, output_size)
                    )

    def forward(self, x):
        return self.model(x)

    def generate(self, batch_size):
        z = utils.cuda(Variable(torch.randn(batch_size, self.noise_size)))
        return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
                        ReLULayer(input_size, hidden_size),
                        ReLULayer(hidden_size, hidden_size),
                        ReLULayer(hidden_size, hidden_size),
                        nn.Linear(hidden_size, output_size),
                        # nn.Sigmoid()  # No sigmoid?
                    )

    def forward(self, x):
        return self.model(x)
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 100 # Batch size
TESTING = False

class TrainGAN(object):
    """docstring for TrainGAN
    made this so we can just call update_iter while keeping the states
    """
    def __init__(self, exp_dir , gan_config, model, task_fs=None):
        """
        Inputs:
            exp_dir: working directory (e.g. where to save logs/results)
            DIM:     hidden width for G/D
            ZDIM:    input dim to G
            train_gen:  generator for training data
            model:   instance of target model
            gan_mode: 'wgan-gp', 'lsgan' ... see self.train_gan
            ITERS:    max iters
            task_fs: list of evaluation tasks

        """
        super(TrainGAN, self).__init__()
        ## aux needed for loss
        tfv_zeros = Variable(torch.zeros(BATCH_SIZE), requires_grad=False).type(torch.cuda.LongTensor)
        tfv_ones = Variable(torch.ones(BATCH_SIZE), requires_grad=False).type(torch.cuda.LongTensor)
        wgan_clip = .01
        LAMBDA = 10
        ## loss
        dic_f_g_loss = {
            'lsgan': lambda D_fake_output, **kwargs:0.5 * torch.mean((D_fake_output - 1) ** 2),
            'dcgan': lambda D_fake_output, **kwargs:F.cross_entropy(D_fake_output, tfv_ones),
            'wgan' : lambda D_fake_output, **kwargs: -torch.mean(D_fake_output),
            'wgan-gp':lambda D_fake_output, **kwargs: -torch.mean(D_fake_output),
        }

        dic_f_d_loss = {
            'lsgan': lambda D_real_output, D_fake_output, **kwargs: 0.5 * (torch.mean((D_real_output - 1) ** 2) + torch.mean(D_fake_output ** 2)),
            'dcgan': lambda D_real_output, D_fake_output, **kwargs: 0.5 * (F.cross_entropy(D_real_output, tfv_ones) + F.cross_entropy(D_fake_output, tfv_zeros)),
            'wgan' : lambda D_real_output, D_fake_output, **kwargs: torch.mean(D_fake_output) - torch.mean(D_real_output),
            'wgan-gp' : lambda D_real_output, D_fake_output, **kwargs: torch.mean(D_fake_output) - torch.mean(D_real_output),
        }

        output_sizes = defaultdict(lambda:1)
        output_sizes['dcgan'] = 2
        
        ## Input
        self.exp_dir   = exp_dir      
        self.DIM       = gan_config['gan_dim']  
        self.ZDIM      = gan_config['zdim']  
        self.OUTPUT_DIM = gan_config['output_dim']
        self.BATCH_SIZE= gan_config['batch_size']
        self.CRITIC_ITERS= gan_config['critic_iters']
        self.model     = model  
        self.gan_mode  = gan_config['mode']
        self.ITERS     = gan_config['iters']
        self.task_fs   = task_fs
        ## Aux
        self.tfv_zeros = tfv_zeros
        self.tfv_ones = tfv_ones
        self.wgan_clip = wgan_clip
        self.LAMBDA = LAMBDA
        self.dic_f_d_loss = dic_f_d_loss
        self.dic_f_g_loss = dic_f_g_loss
        self.output_sizes = output_sizes
        self.GAN_EXPERIMENT_DIR = os.path.abspath(GAN_EXPERIMENT_DIR)
    def init_gan(self):
        OUTPUT_DIM = self.OUTPUT_DIM
        ZDIM = self.ZDIM 
        DIM = self.DIM
        output_sizes = self.output_sizes
        gan_mode = self.gan_mode

        ####
        ZDIM = ZDIM or 32
        
        # Train loop
        lr = 0.0001
        G = Generator(noise_size=ZDIM, hidden_size=DIM, output_size=OUTPUT_DIM)  # 64 for Baby MNIST
        D = Discriminator(input_size=OUTPUT_DIM, hidden_size=DIM, output_size=output_sizes[gan_mode])
        G.cuda()
        D.cuda()
        G_optimizer = optim.Adam(G.parameters(), lr=lr)
        D_optimizer = optim.Adam(D.parameters(), lr=lr)

        self.lr = lr
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

    def train_gan(self, train_gen):
        BATCH_SIZE = self.BATCH_SIZE
        ZDIM = self.ZDIM
        G = self.G
        model = self.model

        G_losses = []
        D_losses = []
        gen = gan_utils.inf_train_gen(train_gen)
        for iteration in range(ITERS):
            data = next(gen)
            gloss, dloss = self.update_iter(iteration, data)
            G_losses.append(gloss)
            D_losses.append(dloss)

            if (iteration > 0) and (iteration % 100 == 0) and task_fs is not None:
                sample_dics = self.get_samples(10 * self.BATCH_SIZE)
                for task_f in task_fs:
                    task_f(sample_dics, iteration)
        return G
    def get_samples(self, N):
        BATCH_SIZE = self.BATCH_SIZE
        ZDIM = self.ZDIM
        G = self.G
        model = self.model

        sample_params = None
        for _ in range(N//BATCH_SIZE+1):
            z = torch.randn(BATCH_SIZE, ZDIM)
            z = Variable(z.cuda())
            sp = G(z).data.cpu().numpy()
            sample_params = sp if sample_params is None else\
                            np.vstack([sample_params, sp])
        sample_dics = utils.prepare_torch_dicts(sample_params[:N], model)
        return sample_dics


    def update_iter(self, iteration, idata):
        G = self.G
        D = self.D
        BATCH_SIZE = self.BATCH_SIZE
        ZDIM = self.ZDIM
        gan_mode = self.gan_mode
        dic_f_g_loss = self.dic_f_g_loss
        dic_f_d_loss = self.dic_f_d_loss
        G_optimizer = self.G_optimizer
        D_optimizer = self.D_optimizer
        exp_dir = self.exp_dir
        GAN_EXPERIMENT_DIR = self.GAN_EXPERIMENT_DIR

        ## go to experiment directory
        if exp_dir: ## i.e. not None
            abs_exp_dir = os.path.join(GAN_EXPERIMENT_DIR,exp_dir)
            if not os.path.exists(os.path.abspath(abs_exp_dir)):
                os.makedirs(os.path.abspath(abs_exp_dir))
            if os.path.abspath(os.curdir) != os.path.abspath(abs_exp_dir):
                os.chdir(os.path.abspath(abs_exp_dir))
        G_loss, D_loss = None, None
        # Train generator
        if iteration > 0:
            # Train generator G
            # -----------------
            G.zero_grad()

            z = torch.randn(BATCH_SIZE, ZDIM)
            z = Variable(z.cuda())

            generated_data = G(z)
            D_fake_output = D(generated_data)
            G_loss = dic_f_g_loss[gan_mode](D_fake_output)
            G_loss.backward()
            G_optimizer.step()

        if gan_mode == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS

        # Train discriminator D
        for i in range(disc_iters):
            data = Variable(torch.from_numpy(idata).cuda())

            D.zero_grad()

            D_real_output = D(data)

            z = torch.randn(BATCH_SIZE, ZDIM)
            z = Variable(z.cuda())
            generated_data = G(z)

            D_fake_output = D(generated_data)
            D_loss = dic_f_d_loss[gan_mode](D_real_output, D_fake_output)
            if gan_mode=='wgan-gp':
                def gradient_penalty(x, y, f):
                    """
                    href:
                    https://github.com/LynnHo/Pytorch-WGAN-GP-DRAGAN-Celeba/blob/master/train_celeba_wgan_gp.py
                    """
                    # interpolation
                    shape = [x.size(0)] + [1] * (x.dim() - 1)
                    alpha = Variable(torch.rand(shape)).cuda()
                    z = x + alpha * (y - x)

                    # gradient penalty
                    # z = Variable(z, requires_grad=True).cuda()
                    o = f(z)
                    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
                    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

                    return gp
                gp = gradient_penalty(data, generated_data, D)
                D_loss += LAMBDA*gp
            D_loss.backward()
            D_optimizer.step()

            if gan_mode=='wgan':
                for p in D.parameters():
                    p.data.clamp_(-wgan_clip, wgan_clip)
        return G_loss.data[0] if G_loss is not None else 0, D_loss.data[0]

if __name__ == '__main__':
    try:
        arguments = docopt(__doc__)
    except Exception as e: ## TODO: tmp using this to enable calling from another file
        print('Error blocked: {}'.format(e))
        arguments = {
            '<src_dir>':'CE.fc1-100-X-sgld-baby-X-babymnist-1000@2017-12-25',
            '<f_opt_config>':'opt/gan-config/babymnist1.yaml',
            '--cuda': True
        }
    src_dir = arguments['<src_dir>']


    opt_config = gan_utils.opt_load_configuration(arguments['<f_opt_config>'], None)
    TASK = opt_config['task']

    print("TASK = {}".format(TASK))
    print("N_TRAIN = {}".format(opt_config['n_train']))

    DATA_DIR = './data'
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in gan.py!')

    exp_dir_prefix = ''
    MODE = opt_config['mode']
    DIM = opt_config['gan_dim']
    exp_dir = exp_dir_prefix+'_'+TASK+'-'+MODE+'-'+'%g'%opt_config['n_train'] + '%g'%DIM
    ZDIM = opt_config['zdim']  # The noise dimension of the generator
    LAMBDA = opt_config['lambda'] # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = opt_config['critic_iters'] # How many critic iterations per generator iteration
    BATCH_SIZE = opt_config['batch_size'] # Batch size
    ITERS = opt_config['iters'] # How many generator iterations to train for
    TESTING = False

    gan_exp_dir = os.path.join(GAN_EXPERIMENT_DIR, exp_dir)
    if not os.path.exists(gan_exp_dir):
        os.makedirs(gan_exp_dir)

    #### validation helpers
    model = gan_utils.model_load_configuration(arguments)
    if arguments['--cuda']:
        model.cuda()  # Move the model onto the GPU

    if TASK == 'toy2d':
        validater = gan_utils.EvalToy2d(model)
        task_fs = [validater.toy2d_validate]
    elif TASK == 'babymnist':
        validater = gan_utils.EvalBabyMNIST(model, opt_config, arguments['--cuda'])
        task_fs = [validater.babymnist_validate, validater.babymnist_ood]
    elif TASK == 'mnist':  # TODO: Incorporate support for FashionMNIST
        validater = gan_utils.EvalMNIST(model, opt_config, arguments['--cuda'])
        task_fs = [validater.mnist_validate, validater.mnist_ood]
    else:
        raise NotImplementedError()
    #### End of validation helpers

    train_gen = data_generator(src_dir, BATCH_SIZE, N_TRAIN=opt_config['n_train'])
    opt_config['output_dim'] = list(train_gen())[0].shape[1]
    with open(os.path.join(gan_exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(opt_config, f)
    obj_traingan = TrainGAN(exp_dir, opt_config, model, task_fs)
    obj_traingan.init_gan()
    try:
        generator = obj_traingan.train_gan(train_gen)
    except KeyboardInterrupt:
        print('Breaking training early')

    print('Saving Generator...')
    torch.save(generator.state_dict(), 'generator.pt')



