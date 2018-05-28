
"""active_learning.py
Usage:
    active_learning.py <acq_crit> <f_model_config> <f_opt_config> <dataset> [--apd_gan=<gan_config>] [--prefix=<p>] [--ce] [--db] [--cuda] [--deter] [--warm] [--mc_dropout_passes=<passes>]
    active_learning.py -r <exp_name> <idx> [--test]

Arguments:

Options:
    --mc_dropout_passes=<passes> The number of MC dropout passes to perform at test time. If 0, don't use MC dropout. [default: 0].
    --prefix=<p> Prefix for experiment name [default:]
    --apd=<gan_config> if using online apd (e.g. opt/gan-config/babymnist-dcgan.yaml) [default:]

Example:
    python active_learning.py entropy model/config/fc1-mnist-100.yaml opt/config/sgld-mnist-1.yaml mnist --cuda

    Available acquisition functions:
        * random
        * entropy
        * bald

    # python active_learning.py model/config/fc1-100.yaml opt/config/nsgd-bdk.yaml babymnist --ce --cuda

Options:
"""
from __future__ import division

import os
import pdb
from tqdm import tqdm
import pickle as pickle
from docopt import docopt

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# Local imports
import utils
from model.fc import *
from model.cnn import *
from opt.loss import *
from opt.nsgd import NoisedSGD
from train_new import (load_configuration,
                       update_LearningRate,
                       check_point,
                       posterior_sampling,
                       _flatten_npyfy)
## GAN related
import gan_utils
from gan_pytorch import *

def update_train_pool_sets(trainset, poolset, selected_idxs, cat=True):
    ## update train/pool set given indices
    poolset_idxs = np.setdiff1d(np.arange(len(poolset.train_labels)), selected_idxs)  # All poolset indexes that are not added to the trainset
    # update dataset objects
    if cat:
        # Type issue:
        # the default type of dataset.train_data is ByteTensor
        # the indexing object cannot be ByteTensor
        # So, convert it to numpy, and then convert it back
        trainset.train_data =   np.concatenate([trainset.train_data.numpy(), poolset.train_data.numpy()[selected_idxs]], 0)
        trainset.train_labels = np.concatenate([trainset.train_labels.numpy(), poolset.train_labels.numpy()[selected_idxs]], 0)

        poolset.train_data = torch.ByteTensor(poolset.train_data)
        poolset.train_labels = torch.LongTensor(poolset.train_labels)
        trainset.train_data = torch.ByteTensor(trainset.train_data)
        trainset.train_labels = torch.LongTensor(trainset.train_labels)

    else:
        trainset.train_data =   poolset.train_data.numpy()[selected_idxs]
        trainset.train_labels = poolset.train_labels.numpy()[selected_idxs]
        trainset.train_data = torch.ByteTensor(trainset.train_data)
        trainset.train_labels = torch.LongTensor(trainset.train_labels)

    poolset.train_data =   torch.ByteTensor(poolset.train_data.numpy()[poolset_idxs])
    poolset.train_labels = torch.LongTensor(poolset.train_labels.numpy()[poolset_idxs])


def acq_random(model, poolset, poolloader, acq_size, arguments, opt_config):
    idxs = np.arange(len(poolset.train_labels))
    np.random.shuffle(idxs)
    return idxs[:acq_size]


def acq_entropy(model, poolset, poolloader, acq_size, arguments, opt_config):
    Hs = []
    model.eval()
    for _, (inputs, _) in enumerate(poolloader):
        if arguments['--cuda']:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)

        # Deteriministic
        if  arguments['--deter']:
            outputs = model.forward(inputs)
            sm = F.softmax(outputs)
        elif arguments['--mc_dropout_passes']:
            sm = utils.mc_dropout_expectation(model, inputs, keep_samples=False, passes=opt_config['test_sample_batch_size'])
        # Bayesian Prediction
        else:
            sm = utils.posterior_expectation(model, inputs, sample_size=opt_config['test_sample_batch_size'])
        
        Hs += list(utils.f_entropy(sm.data.cpu().numpy()))
    model.train()
    return np.argsort(Hs)[::-1][:acq_size]


def acq_bald(model, poolset, poolloader, acq_size, arguments, opt_config):
    # Deteriministic
    if arguments['--deter']:
        return acq_random(model, poolset, poolloader, acq_size, arguments)

    balds = []
    model.eval()
    for _, (inputs, _) in enumerate(poolloader):
        if arguments['--cuda']:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        sm, sample_sms = utils.posterior_expectation(model, inputs, keep_samples=True, sample_size=opt_config['test_sample_batch_size'])
        balds += list(utils.f_bald([sm,
                     sample_sms]))

    model.train()
    return np.argsort(balds)[::-1][:acq_size]


acq_dict = {
    'acq_bald': acq_bald,
    'acq_entropy': acq_entropy,
    'acq_random': acq_random
}


def main(arguments):



    ## active learning configuration
    # fixed (i.e. from Gal'17)
    initial_data_per_class = 2
    acq_size = 10
    validation_set_size = 100
    # can tune
    num_acq = 50
    iterations_per_acq = 3000
    acq_crit = 'acq_' + arguments['<acq_crit>']



    name_dataset = arguments['<dataset>']
    # Load arguments: Module, Optimizer, Loss_function
    model, optimizer, Loss, exp_name, model_config, opt_config = load_configuration(arguments, name_dataset)
    num_iterations = iterations_per_acq

    if arguments['--apd_gan']:
        ####
        #### GAN CONFIG
        ####
        gan_config = gan_utils.opt_load_configuration(arguments['--apd_gan'], None)
        ### compute OUTPUT_DIM
        posterior_sampling(1, model, 1)
        gan_config['output_dim'] = _flatten_npyfy(model.posterior_samples).shape[1]
        model.posterior_samples = []
        model.posterior_weights = []
        ###
        obj_traingan = TrainGAN(None, gan_config, model, [])
        obj_traingan.init_gan()
        ###
        gan_bs = gan_config['batch_size'] # Batch size
        gan_inp_buffer = []
        gan_iter = 0
        
    default_config = {
        'batcher_kwargs':{'batch_size': 100},
        'burnin_iters': 500,
        'sample_size': 10,
        'sample_interval': 20,
        'validation_interval': 2000,
        'ood_scale': 5,
        'variance_monitor_interval': 50,
        'ood_datasets': ['notMNIST']
    }
    
    # Fill in default configuration for keys that are not overwritten by the config file
    for key in default_config:
        if key not in opt_config:
            opt_config[key] = default_config[key]

    # Fill in default n_anom value, if not overwritten by the config file
    if 'n_anom' not in opt_config:
        if name_task.split('-')[0] in ['mnist', 'fashion']:
            opt_config['n_anom'] = 2000
        elif name_task.split('-')[0] == 'babymnist':
            opt_config['n_anom'] = 300

    exp_name = '{}{}{}-{}'.format('D' if arguments['--deter'] else 'B',
                                  'W' if arguments['--warm'] else 'C',
                                  arguments['<acq_crit>'],
                                  exp_name)

        # Set up
    batch_size = 100
    # alpha_threshold = .5
    burnin_iters = opt_config['burnin_iters']
    sample_size = 1 if arguments['--deter'] else opt_config['sample_size']
    sample_interval = opt_config['sample_interval']
    validation_interval = 500
    variance_monitor_interval = 50
    is_collecting = False
    pool_batch_size = 100

    print('Experiment name: {}'.format(exp_name))

    log_folder = os.path.join('./logs', exp_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # Load DataSet
    # trainLoader automatically generate training_batch
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if name_dataset == 'babymnist':
        trainset = BabyMnist(train=True, transform=transform)
        poolset = BabyMnist(train=True, transform=transform)
        testset = BabyMnist(train=False, transform=transform)
    if name_dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        poolset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # trainloader, valloader, testloader, name_dataset = utils.get_dataloader(name_dataset, batch_size)


    # Tensorboard
    # # Set tensorboard_monitor monitors
    # loss_monitor = Monitor('cross_entropy')
    # accuracy_monitor = Monitor('accuracy')
    # learning_rate_monitor = Monitor('learning_rate')
    # variance_monitor = Monitor('variance')
    # Initial tensorboard
    # sess, train_writer, point_writer, posterior_writer, mcdropout_writer = initial_tensorboard(exp_name)

    acq_test_acc = []
    acq_anom_detection = []

    ood_data = utils.load_ood_data(name_dataset, opt_config)

    ################
    ### Training ###
    ################
    for acq_idx in range(num_acq):
        print("Acquisition Iter {}/{}".format(acq_idx+1, num_acq))

        if acq_idx == 0:  ### initial setup
            ### create a class-balanced initial training set
            init_trainset_idxs = [[] for i in range(10)]
            counter = 0
            for (idx, label) in enumerate(poolset.train_labels):
                if len(init_trainset_idxs[label]) < initial_data_per_class:
                    init_trainset_idxs[label].append(idx)
                    counter += 1
                if counter == initial_data_per_class * 10:
                    break
            trainset_idxs = np.array(init_trainset_idxs).ravel()
            cat = False

        else:  ### acquire!
            if arguments['--apd_gan']:
                print('COMPRESSING GAN')
                gan_inp_buffer = _flatten_npyfy(model.posterior_samples)
                for _ in range(1000):
                    idxs = np.arange(len(gan_inp_buffer))
                    np.random.shuffle(idxs)
                    obj_traingan.update_iter(gan_iter, np.array(gan_inp_buffer)[idxs[:gan_bs]])
                print('DE-COMPRESSING GAN')
                ## Model
                gmodel = eval(model_config['name'])(**model_config['kwargs'])
                gmodel = utils.cuda(gmodel, arguments)
                ## get the samples
                gmodel.posterior_samples = obj_traingan.get_samples(len(model.posterior_samples))
                gmodel.posterior_weights = [1 for _ in range(len(gmodel.posterior_samples))]
                ## replace
                model = gmodel
            # acquire!
            trainset_idxs = acq_dict[acq_crit](model, poolset, poolloader, acq_size, arguments, opt_config)
            cat = True

        update_train_pool_sets(trainset, poolset, trainset_idxs, cat)
        ## create loaders
        print("Current Training Set Size: {}".format(len(trainset.train_labels)))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.train_labels), shuffle=True, num_workers=2)
        poolloader = torch.utils.data.DataLoader(poolset, batch_size=pool_batch_size, shuffle=False, num_workers=2)

        ## cold/warm-start the model
        if acq_idx > 0:
            if not arguments['--warm']:
                model = eval(model_config['name'])(**model_config['kwargs'])
                if arguments['--cuda']:
                    model.type(torch.cuda.FloatTensor)
                # Optimizer
                optimizer = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])
            else:
                num_iterations = int(arguments['<fine_tune_iter>'])

        # preproc train data
        data = iter(trainloader).next()
        inputs, labels = data
        if arguments['--cuda']:
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        ## adjust optimizer 
        if opt_config['name'] == 'NoisedSGD':
            optimizer.correction = float(len(trainset.train_labels)) *1000./2.#...
        for iteration in range(num_iterations):
            # Update learning rate
            learning_rate = update_LearningRate(optimizer, iteration, opt_config)
            # learning_rate_monitor.record_tensorboard(learning_rate, iteration, sess, train_writer)

            # Inference
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            training_loss = F.cross_entropy(outputs, labels)
            training_loss.backward()  # Computes gradients of the model parameters wrt the loss

            prob_outputs = F.softmax(outputs)
            training_predictions = prob_outputs.data.cpu().numpy().argmax(1)

            accuracy = utils.inference_accuracy(training_predictions, labels)
            # accuracy_monitor.record_tensorboard(accuracy, iteration, sess, train_writer)

            optimizer.step()

            # monitor Variance
            if opt_config['name'] == 'NoisedSGD' and is_collecting == False and iteration % variance_monitor_interval == 0:
                # is_collecting = is_into_langevin_dynamics(model, outputs,  gradient_data, optimizer, learning_rate,alpha_threshold,iteration,sess, train_writer)
                is_collecting = iteration >= burnin_iters
            if is_collecting and iteration % sample_interval == 0:
                posterior_sampling(sample_size, model, learning_rate)


            # Validation
            if (iteration % validation_interval == 0):

                point_accuracy = []
                point_loss = []
                posterior_accuracy = []
                posterior_loss = []
                posterior_flag = 0

                test_inputs_anomoly_detection = []

                for i, data in enumerate(testloader, 0):

                    # Load data
                    # data for inference
                    test_inputs, test_labels = data
                    if arguments['--cuda']:
                        test_inputs = test_inputs.type(torch.cuda.FloatTensor)
                        test_labels = test_labels.cuda()
                    test_inputs, test_labels = Variable(test_inputs, volatile=True), Variable(test_labels, volatile=True)

                    model.eval()
                    # Prediction
                    # Point Prediction
                    point_outputs = model.forward(test_inputs)
                    point_loss_batch = F.cross_entropy(point_outputs, test_labels)
                    point_loss.append(point_loss_batch.data[0])

                    point_predictions = Loss.inference_prediction(point_outputs)
                    point_accuracy_batch = utils.inference_accuracy(point_predictions, test_labels)
                    point_accuracy.append(point_accuracy_batch)

                    # Bayesian Prediction
                    if len(model.posterior_samples) > 0 and opt_config['name'] == 'NoisedSGD':

                        posterior_flag = 1
                        posterior_outputs = utils.posterior_expectation(model, test_inputs, sample_size=opt_config['test_sample_batch_size'])
                        posterior_loss_batch = F.nll_loss(torch.log(posterior_outputs), test_labels.cpu())
                        posterior_loss.append(posterior_loss_batch.data[0])

                        posterior_predictions = Loss.inference_prediction(posterior_outputs)
                        posterior_accuracy_batch = utils.inference_accuracy(posterior_predictions, test_labels)
                        posterior_accuracy.append(posterior_accuracy_batch)

                    
                # record prediction result
                point_accuracy = np.mean(point_accuracy)
                point_loss = np.mean(point_loss)
                # accuracy_monitor.record_tensorboard(point_accuracy, iteration, sess, point_writer)
                # loss_monitor.record_tensorboard(point_loss, iteration, sess, point_writer)

                if posterior_flag == 1:
                    posterior_accuracy = np.mean(posterior_accuracy)
                    posterior_loss = np.mean(posterior_loss)
                    # accuracy_monitor.record_tensorboard(posterior_accuracy, iteration, sess, posterior_writer)
                    # loss_monitor.record_tensorboard(posterior_loss, iteration, sess, posterior_writer)
                    # acq_test_acc.append(posterior_accuracy)
                # else:
                #     acq_test_acc.append(point_accuracy)  # Is this right?
                #     pdb.set_trace()

                print('It: {} | Trn Loss: {}| Trn acc: {} | Tst point acc: {} | Tst posterior acc: {}'.format(iteration, training_loss.data[0],accuracy, point_accuracy, posterior_accuracy))


            # Termination
            if iteration == num_iterations - 1:
                # loss_monitor.save_result_numpy(log_folder)
                print("This iteration of Active Learning finished {}/{}".format(acq_idx+1, num_acq))
                print(accuracy, point_accuracy, posterior_accuracy)
                acq_test_acc.append(posterior_accuracy)
                # pdb.set_trace()
                # acq_anom_detection.append((auroc, n_aupr, ab_aupr))

            idx = iteration

            # check_point(model, optimizer, iteration, exp_name)
        np.save(os.path.join(log_folder, 'acq_acc'), np.array(acq_test_acc))
        np.save(os.path.join(log_folder, 'acq_anom'), np.array(acq_anom_detection))
        pickle.dump(arguments, open(os.path.join(log_folder, 'args.pkl'), 'wb'))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    main(arguments)
