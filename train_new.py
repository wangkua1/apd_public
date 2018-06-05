"""train_new.py

Usage:
    train_new.py <f_model_config> <f_opt_config> <dataset> [--dont-save] [--db] [--cuda] [--test] [--mc_dropout_passes=<passes>] [--apd_gan=<gan_config>] [--apd=<apd_config>] [--prefix=<p>]
    train_new.py <f_model_config> <f_opt_config> <dataset> [--dont-save] [--db] [--cuda] [--test] [--mc_dropout_passes=<passes>] [--prefix=<p>]
    train_new.py -r <exp_name> <idx> [--test]

Options:
    --mc_dropout_passes=<passes> The number of MC dropout passes to perform at test time. If 0, don't use MC dropout. [default: 0].
    --prefix=<p> Prefix for experiment name [default:]
    --apd=<gan_config> if using online apd (e.g. opt/gan-config/babymnist-dcgan.yaml) [default:]

Arguments:

Example:
    Jan05,2018
    python train_new.py model/config/fc1-100.yaml opt/config/sgld-baby.yaml babymnist-1000 --cuda --apd_gan opt/gan-config/babymnist-wgan-gp.yaml --apd opt/apd-config/vanilla_apd.yaml
    python train_new.py model/config/fc1-mnist-100.yaml opt/config/sgld-mnist-1.yaml mnist-50000 --cuda

    python train_new.py model/config/cnn-globe.yaml opt/config/sgld-mnist-1.yaml mnist-50000 --cuda
"""
import matplotlib as mtl
mtl.use('Agg')
import os
import pdb
import copy
import yaml
import shutil
import datetime
import pickle as pkl
# import tabulate
from docopt import docopt

from tensorboard_monitor.configuration import *
from tensorboard_monitor.monitor import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# from tqdm import tqdm
import pickle as pickle

# Local imports
import utils
from opt.loss import *
from model.fc import *
from model.cnn import *
from opt.nsgd import NoisedSGD

torch.backends.cudnn.enabled = False
## GAN related
import gan_utils
from gan_pytorch import *


def load_configuration(arguments, name_dataset):
    if arguments['-r']:
        exp_name = arguments['<exp_name>']
        f_model_config = 'model/config/' + exp_name[exp_name.find(':') + 1:].split('-X-')[0] + '.yaml'
        f_opt_config = 'opt/config/' + exp_name[exp_name.find(':') + 1:].split('-X-')[1] + '.yaml'
        old_exp_name = exp_name
        exp_name += '_resumed'
    else:
        f_model_config = arguments['<f_model_config>']
        f_opt_config = arguments['<f_opt_config>']
        model_name = os.path.basename(f_model_config).split('.')[0]
        opt_name = os.path.basename(f_opt_config).split('.')[0]
        timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
        data_name = name_dataset
        if arguments['--prefix'] :
            exp_name = '%s:%s-X-%s-X-%s@%s' % (arguments['--prefix'], model_name, opt_name, data_name, timestamp)
        else:
            exp_name = '%s-X-%s-X-%s@%s' % (model_name, opt_name, data_name, timestamp)

    model_config = yaml.load(open(f_model_config, 'rb'))
    opt_config = yaml.load(open(f_opt_config, 'rb'))

    print('\n\n\n\n>>>>>>>>> [Experiment Name]')
    print(exp_name)
    print('<<<<<<<<<\n\n\n\n')

    ## Experiment stuff
    # create exp dir and copy the configs over
    exp_dir_name = './saves/{}'.format(exp_name)
    if not os.path.exists(exp_dir_name):
        os.makedirs(exp_dir_name)
        model_config_fname = os.path.basename(f_model_config)
        opt_config_fname = os.path.basename(f_opt_config)
        shutil.copy(f_model_config, os.path.join(exp_dir_name, model_config_fname))
        shutil.copy(f_opt_config, os.path.join(exp_dir_name, opt_config_fname))

    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    model = utils.cuda(model, arguments)

    ## Optimizer
    opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])

    Loss = CE()

    if arguments['-r']:
        model.load('./saves/%s/model_%s.t7' % (old_exp_name, arguments['<idx>']))
        opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7' % (old_exp_name, arguments['<idx>'])))
        if arguments['--test']:
            raise NotImplementedError()


    # exp_name = 'cifar-cnn-globe-X-sgld-cifar5-X-cifar5-20000@2018-02-06'

    return model, opt, Loss, exp_name, model_config, opt_config


def update_LearningRate(optimizer, iteration, opt_config):
    sd = optimizer.state_dict()
    learning_rate = sd['param_groups'][0]['lr']
    if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == iteration:
        raise ("I didn't not support LR schedule after multi-chain MCMC")
        _, tmp_fac = opt_config['lrsche'].pop(0)
        sd = optimizer.state_dict()
        assert len(sd['param_groups']) == 1
        sd['param_groups'][0]['lr'] *= tmp_fac
        optimizer.load_state_dict(sd)
    if iteration > 0 and 'lrpoly' in opt_config:
        raise ("I didn't not support LR schedule after multi-chain MCMC")
        a, b, g = opt_config['lrpoly']
        sd = optimizer.state_dict()
        step_size = a * ((b + iteration) ** (-g))
        sd['param_groups'][0]['lr'] = step_size
        optimizer.load_state_dict(sd)
    return learning_rate


def check_point(model, optimizer, iteration, exp_name):
    if iteration > 0 and iteration % 1000 == 0:
        name = './saves/%s/model_%i.t7' % (exp_name, iteration)
        print("[Saving to] {}".format(name))

        torch.save(model.state_dict(), name)
        torch.save(optimizer.state_dict(), './saves/%s/opt_%i.t7' % (exp_name, iteration))


def posterior_sampling(sample_size, model, learning_rate):
    model.cpu()
    model.posterior_samples.append(copy.deepcopy(model.state_dict()))
    model.cuda()
    model.posterior_weights.append(learning_rate)
    if len(model.posterior_samples) > sample_size:
        del model.posterior_samples[0]
        del model.posterior_weights[0]


#########################
### Anomaly Detection ###
#########################
def run_test_mc_dropout(model, dataloader, arguments):
    # To be used on val and test sets
    was_training = model.training
    model.train()  # Important: keep train mode on for MC dropout

    mc_posterior_accuracy_list = []
    mc_posterior_loss_list = []

    for data in dataloader:

        inputs, labels = data
        inputs = utils.cuda(inputs, arguments)
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        mean_probs = utils.mc_dropout_expectation(model, inputs, keep_samples=False, passes=int(arguments['--mc_dropout_passes']))

        mean_predictions = mean_probs.data.cpu().numpy().argmax(1)
        mc_posterior_accuracy_batch = utils.inference_accuracy(mean_predictions, labels)
        mc_posterior_loss_batch = F.nll_loss(torch.log(mean_probs.cpu()), labels)
        mc_posterior_accuracy_list.append(mc_posterior_accuracy_batch)
        mc_posterior_loss_list.append(mc_posterior_loss_batch.data.cpu().numpy())

    mc_posterior_accuracy = np.mean(mc_posterior_accuracy_list)
    mc_posterior_loss = np.mean(mc_posterior_loss_list)

    if not was_training:
        model.eval()
    return mc_posterior_accuracy, mc_posterior_loss.data[0]


def get_dataloader(task, batch_size):
    size_trainset = int(task.split('-')[1])
    name_dataset = task.split('-')[0]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if name_dataset == 'babymnist':
        # Train set
        trainset = BabyMnist(train=True, transform=transform)
        trainset.train_data = trainset.train_data[:size_trainset, :, :, :]
        trainset.train_labels = trainset.train_labels[:size_trainset]
        # Validation set
        valset = BabyMnist(train=True, transform=transform)
        valset.train_data = valset.train_data[size_trainset:, :, :, :]
        valset.train_labels = valset.train_labels[size_trainset:]
        # Test set
        testset = BabyMnist(train=False, transform=transform)
    elif name_dataset == 'mnist':
        # Train set
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainset.train_data = trainset.train_data[:size_trainset, :, :]
        trainset.train_labels = trainset.train_labels[:size_trainset]
        # Validation set
        valset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset.train_data = valset.train_data[size_trainset:, :, :]
        valset.train_labels = valset.train_labels[size_trainset:]
        # Test set
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name_dataset == 'fashion':
        # Train set
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainset.train_data = trainset.train_data[:size_trainset, :, :]
        trainset.train_labels = trainset.train_labels[:size_trainset]
        # Validation set
        valset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valset.train_data = valset.train_data[size_trainset:, :, :]
        valset.train_labels = valset.train_labels[size_trainset:]
        # Test set
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif name_dataset == 'cifar10':
        # Train set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainset.train_data = trainset.train_data[:size_trainset, :, :, :]
        trainset.train_labels = trainset.train_labels[:size_trainset]
        # Validation set
        valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valset.train_data = valset.train_data[size_trainset:, :, :]
        valset.train_labels = valset.train_labels[size_trainset:]
        # Test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader, name_dataset


def evaluate(model, testloader, posterior_flag, Loss, opt_config):
    model.eval()
    posterior_weights = model.posterior_weights
    posterior_samples = model.posterior_samples


    point_accuracy = []
    point_loss = []
    posterior_accuracy = []
    posterior_loss = []

    for i, data in enumerate(testloader, 0):

        # Load data
        # data for inference
        test_inputs, test_labels = data
        test_inputs, test_labels = utils.cuda((test_inputs, test_labels), arguments)
        test_inputs, test_labels = Variable(test_inputs, volatile=True), Variable(test_labels, volatile=True)

        # Prediction
        # Point Prediction
        point_outputs = model.forward(test_inputs)
        point_loss_batch = F.cross_entropy(point_outputs, test_labels)
        point_loss.append(point_loss_batch.data[0])

        point_predictions = Loss.inference_prediction(point_outputs)
        point_accuracy_batch = utils.inference_accuracy(point_predictions, test_labels)
        point_accuracy.append(point_accuracy_batch)

        # Bayesian Prediction
        if posterior_flag:
            # posterior_outputs = utils.posterior_expectation(model, test_inputs)
            posterior_outputs = utils.posterior_expectation(model, test_inputs, keep_samples=False, use_mini_batch=opt_config['batch_size'])
            # posterior_loss_batch = Loss.nll(torch.log(posterior_outputs), test_labels)
            posterior_loss_batch = F.nll_loss(torch.log(posterior_outputs), test_labels.cpu())
            posterior_loss.append(posterior_loss_batch.data[0])

            posterior_predictions = Loss.inference_prediction(posterior_outputs)
            posterior_accuracy_batch = utils.inference_accuracy(posterior_predictions, test_labels)
            posterior_accuracy.append(posterior_accuracy_batch)

    model.train()

    # record prediction result
    point_accuracy = np.mean(point_accuracy)
    point_loss = np.mean(point_loss)

    if posterior_flag == 1:
        posterior_accuracy = np.mean(posterior_accuracy)
        posterior_loss = np.mean(posterior_loss)

    return point_accuracy, point_loss, posterior_accuracy, posterior_loss


def _flatten_npyfy(posterior_samples):
    ret = []
    for sample in posterior_samples:
        item = []
        for seq in sample:
            for p in seq.values():
                item.append(p.cpu().numpy().ravel())
        ret.append(item)

    return np.array([np.concatenate(item) for item in ret])


def main(arguments):
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
    name_task = arguments['<dataset>']
  # Load argumentts: Module, Optimizer, Loss_function
    model, optimizer, Loss, exp_name, model_config, opt_config = load_configuration(arguments, name_task)
    models = [model]
    #####
    #apd#
    #####
    if arguments['--apd']:
        if not arguments['--apd_gan']:
            raise Exception("specify a gan config")
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
        ####
        #### apd CONFIG
        ####
        apd_config = yaml.load(open(arguments['--apd'], 'rb'))
        ## if any of the following is 0, make it =gan_bs
        apd_config['apd_buffer_size'] = apd_config['apd_buffer_size'] or gan_bs
        apd_config['T_sgld'] = apd_config['T_sgld'] or gan_bs
        apd_config['N_chains'] = apd_config['N_chains'] or gan_bs
        if apd_config['N_chains'] > 1:
            opts = [optimizer]
            for _ in range(apd_config['N_chains'] - 1):
                ### TODO: play with model init...
                model = eval(model_config['name'])(**model_config['kwargs'])
                model = utils.cuda(model, arguments)
                models.append(model)
                ###
                opts.append(eval(opt_config['name'])(model.parameters(), **opt_config['kwargs']))




    #####
    #apd-
    #####

    if arguments['--test']:
        num_max_iteration = 0  # Don't do any training iterations, just jump to the test code
    else:
        num_max_iteration = opt_config['max_train_iters']

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

    batch_size = opt_config['batcher_kwargs']['batch_size']
    burnin_iters = opt_config['burnin_iters']
    sample_size = opt_config['sample_size']
    sample_interval = opt_config['sample_interval']
    validation_interval = opt_config['validation_interval']
    ood_scale = opt_config['ood_scale']
    variance_monitor_interval = opt_config['variance_monitor_interval']

    iteration = 0
    is_collecting = False

    total_num_samples_collected = 0

    log_folder = os.path.join('./logs', exp_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # To keep logs of the best results found during training
    # output_log_file = open(os.path.join('output', exp_name + ".txt"), 'w')
    best_test_point_acc = 0
    best_test_mc_acc = 0

    ####################
    ### Load dataset ###
    ####################
    # trainLoader automatically generate training_batch
    trainloader, valloader, testloader, name_dataset = utils.get_dataloader(name_task, batch_size)
    ood_data = utils.load_ood_data(name_dataset, opt_config)

    # Monitors
    monitor = Monitor()
    for k in ['train_loss', 'point_loss',   'bayesian_loss',    'apd_loss' ,'mcdrop_loss',
              'train_acc',  'point_acc',    'bayesian_acc',     'apd_acc' , 'mcdrop_acc',
                            'point_auroc',  'bayesian_auroc',   'apd_auroc','mcdrop_auroc',
                            'point_aupr+',  'bayesian_aupr+',   'apd_aupr+','mcdrop_aupr+',
                            'point_aupr-',  'bayesian_aupr-',   'apd_aupr-','mcdrop_aupr-',
                            'bbald_auroc','bbald_aupr+','bbald_aupr-',
                            'abald_auroc','abald_aupr+','abald_aupr-']: monitor.dict_of_traces[k] = []


    ################
    ### Training ###
    ################

    # # At any point you can hit Ctrl+C to break out of training early, and proceed to run on test data.
    try:
        # for iteration in range(num_max_iteration):
        while iteration < num_max_iteration:
            for i, data in enumerate(trainloader, 0):

                inputs, labels = utils.cuda(data, arguments)
                inputs, labels = Variable(inputs), Variable(labels)

                # Update learning rate
                learning_rate = update_LearningRate(optimizer, iteration, opt_config)

                # Inference
                optimizer.zero_grad()
                outputs = models[0].forward(inputs)

                training_loss = F.cross_entropy(outputs, labels)
                training_loss.backward()  # Computes gradients of the model parameters wrt the loss

                monitor.record_matplot(training_loss.data[0], iteration, 'train_loss')

                prob_outputs = F.softmax(outputs)
                training_predictions = prob_outputs.data.cpu().numpy().argmax(1)

                accuracy = utils.inference_accuracy(training_predictions, labels)
                monitor.record_matplot(accuracy, iteration, 'train_acc')

                optimizer.step()

                ## run rest of the chains
                if len(models) > 1:
                    for idx in range(1,len(models)):
                        opts[idx].zero_grad()
                        loss = F.cross_entropy(models[idx].forward(inputs), labels)
                        loss.backward()
                        opts[idx].step()

                # Monitor variance
                if (opt_config['name'] == 'NoisedSGD') and (is_collecting == False) and (iteration % variance_monitor_interval == 0):
                    is_collecting = iteration >= burnin_iters
                # Sample MCMC
                if (is_collecting == True) and (iteration % sample_interval == 0):
                    for model in models:
                        posterior_sampling(sample_size, model, learning_rate)

                if len(model.posterior_samples) > 0 and opt_config['name'] == 'NoisedSGD':
                    posterior_flag = 1
                else:
                    posterior_flag = 0

                # Val-Set Evaluation
                if (iteration % validation_interval == 0):
                    ######################
                    ### Classification ###
                    ######################
                    print("Num posterior samples/Model: {}\nNum Chains: {}, Total: {}".\
                                format(len(model.posterior_samples), len(models), len(models)*len(model.posterior_samples) ))
                    # point_accuracy, point_loss, posterior_accuracy, posterior_loss = evaluate(models[1], valloader,  posterior_flag, Loss)
                    point_accuracy, point_loss, posterior_accuracy, posterior_loss = evaluate(models[0], valloader,  posterior_flag, Loss, opt_config)

                    monitor.record_matplot(point_accuracy, iteration, 'point_acc')
                    monitor.record_matplot(point_loss, iteration, 'point_loss')

                    if posterior_accuracy:
                        monitor.record_matplot(posterior_accuracy, iteration, 'bayesian_acc')
                        monitor.record_matplot(posterior_loss, iteration, 'bayesian_loss')
                    if posterior_accuracy:
                        print("It: {:5d} | Acc: {:.4f} | Point Acc: {:.4f} | Posterior Acc: {:.4f}".format(iteration, accuracy, point_accuracy, posterior_accuracy))
                        print("It: {:5d} | Loss: {:.4f} | Point Loss: {:.4f} | Posterior Loss: {:.4f}".format(iteration, training_loss.data[0], point_loss, posterior_loss))
                        sys.stdout.flush()
                    else:
                        print("It: {:5d} | Acc: {:.4f} | Point Acc: {:.4f}".format(iteration, accuracy, point_accuracy))
                        sys.stdout.flush()

                    ##################
                    ### MC-Dropout ###
                    ##################
                    if arguments['--mc_dropout_passes'] and int(arguments['--mc_dropout_passes']) > 0:
                        test_mc_accuracy, test_mc_loss = run_test_mc_dropout(model, valloader, arguments)
                        monitor.record_matplot(test_mc_accuracy, iteration, 'mcdrop_acc')
                        monitor.record_matplot(test_mc_loss, iteration,     'mcdrop_loss')
                        print("MC-Dropout Val Accuracy: {:.4f}".format(test_mc_accuracy))
                        print("MC-Dropout Val Loss: {:.4f}".format(test_mc_loss))
                        sys.stdout.flush()




                    ###########################
                    ### Save Best Val Model ###
                    ###########################
                    # Store the model with the best validation accuracy (either point-accuracy or MC-Dropout accuracy)
                    if arguments['--mc_dropout_passes'] and int(arguments['--mc_dropout_passes']) > 0 and test_mc_accuracy > best_test_mc_acc:
                        best_test_mc_acc = test_mc_accuracy
                        print('Saving model with best validation accuracy!')
                        torch.save(model.state_dict(), os.path.join('saves', exp_name, 'best_mc_model.th'))
                    elif point_accuracy > best_test_point_acc:
                        best_test_point_acc = point_accuracy
                        print('Saving model with best validation accuracy!')
                        torch.save(model.state_dict(), os.path.join('saves', exp_name, 'best_point_model.th'))

                    ####################
                    ### apd Evaluate ###
                    ####################
                    if arguments['--apd']:
                        ## create a new model instance to hold GAN samples
                        ## Model
                        gmodel = eval(model_config['name'])(**model_config['kwargs'])
                        gmodel = utils.cuda(gmodel, arguments)
                        ## get the samples
                        gmodel.posterior_samples = obj_traingan.get_samples(sample_size)
                        gmodel.posterior_weights = [1 for _ in range(len(gmodel.posterior_samples))]

                        ## copied from above, TODO: loop
                        point_accuracy, point_loss, posterior_accuracy, posterior_loss = evaluate(gmodel, valloader,  posterior_flag, Loss, opt_config)

                        # monitor.record_matplot(point_accuracy, iteration, 'point_acc')
                        # monitor.record_matplot(point_loss, iteration, 'point_loss')

                        if posterior_accuracy:
                            monitor.record_matplot(posterior_accuracy,  gan_iter,   'apd_acc')
                            monitor.record_matplot(posterior_loss,      gan_iter,       'apd_loss')
                        if posterior_accuracy:
                            print("GAN It: {:5d} | Acc: {:.4f} | Point Acc: {:.4f} | Posterior Acc: {:.4f}".format(gan_iter, accuracy, point_accuracy, posterior_accuracy))
                        else:
                            print("GAN It: {:5d} | Acc: {:.4f} | Point Acc: {:.4f}".format(gan_iter, accuracy, point_accuracy))

                    #########################
                    ### Anomaly detection ###
                    #########################
                    test_inputs_anomaly_detection = utils.get_anomaly_detection_test_inputs(valloader, opt_config, arguments)
                    test_inputs_anomaly_detection = utils.cuda(test_inputs_anomaly_detection, arguments)
                    ood_data = utils.cuda(ood_data, arguments)


                    # # Bayesian
                    for ood_dataset_name in opt_config['ood_datasets']:
                        print("OOD Dataset: {}".format(ood_dataset_name))

                        cur_ood_data = ood_data[ood_dataset_name]

                        for func_name in opt_config['ood_acq_funcs']:

                            if func_name != 'f_bald':
                                # Non-bayesian
                                normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(
                                    test_inputs_anomaly_detection, cur_ood_data, utils.sm_given_np_data, {'model': model}, f_acq=func_name)
                                print(
                                    "(Non-Bayesian {}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(
                                        func_name, normality_base_rate, auroc, n_aupr, ab_aupr))

                    # Bayesian
                    if posterior_flag == 1:
                        print

                        for ood_dataset_name in opt_config['ood_datasets']:
                            print("OOD Dataset: {}".format(ood_dataset_name))
                            cur_ood_data = ood_data[ood_dataset_name]
                            for func_name in opt_config['ood_acq_funcs']:
                                print("Func name = {}".format(func_name))
                                if func_name == 'f_bald':
                                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                           cur_ood_data,
                                                                                                                           utils.posterior_expectation,
                                                                                                                           {'model': model, 'keep_samples': True, 'use_mini_batch': opt_config['batch_size']},
                                                                                                                           f_acq='f_bald')
                                else:
                                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                           cur_ood_data,
                                                                                                                           utils.posterior_expectation,
                                                                                                                           {'model': model, 'keep_samples': False, 'use_mini_batch': opt_config['batch_size']},
                                                                                                                           f_acq=func_name)
                                print("({}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(func_name.upper(), normality_base_rate, auroc, n_aupr, ab_aupr))



                    # MC-Dropout
                    if arguments['--mc_dropout_passes'] and int(arguments['--mc_dropout_passes']) > 0:
                        print

                        for ood_dataset_name in opt_config['ood_datasets']:
                            print("OOD Dataset: {}".format(ood_dataset_name))
                            cur_ood_data = ood_data[ood_dataset_name]

                            for func_name in opt_config['ood_acq_funcs']:
                                if func_name == 'f_bald':
                                    keep_samples = True
                                else:
                                    keep_samples = False
                                normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                       cur_ood_data,
                                                                                                                       utils.mc_dropout_expectation,
                                                                                                                       {'model': model, 'passes': arguments['--mc_dropout_passes'], 'keep_samples': keep_samples},
                                                                                                                       f_acq=func_name)
                                print("({}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(func_name.upper(), normality_base_rate, auroc, n_aupr, ab_aupr))

                    ################
                    ### apd Anom ###
                    ################
                    if arguments['--apd']:
                        ## create a new model instance to hold GAN samples
                        ## Model
                        gmodel = eval(model_config['name'])(**model_config['kwargs'])
                        gmodel = utils.cuda(gmodel, arguments)
                        ## get the samples
                        gmodel.posterior_samples = obj_traingan.get_samples(sample_size)
                        gmodel.posterior_weights = [1 for _ in range(len(gmodel.posterior_samples))]

                        ## copied from above, TODO: loop
                        for ood_dataset_name in opt_config['ood_datasets']:
                            print("OOD Dataset: {}".format(ood_dataset_name))

                            cur_ood_data = ood_data[ood_dataset_name]
                            normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection, cur_ood_data,  utils.posterior_expectation,
                                                                                                                   {'model': gmodel, 'use_mini_batch': opt_config['batch_size']})
                            print(
                            "(apd) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(
                                normality_base_rate, auroc, n_aupr, ab_aupr))
                            monitor.record_matplot(auroc, gan_iter, 'apd_auroc')
                            monitor.record_matplot(n_aupr, gan_iter, 'apd_aupr+')
                            monitor.record_matplot(ab_aupr, gan_iter, 'apd_aupr-')

                            normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection, cur_ood_data,  utils.posterior_expectation,
                                                                                                                   {'model': gmodel,'keep_samples': True, 'use_mini_batch': opt_config['batch_size']}, f_acq='f_bald')
                            print(
                            "(apd BALD) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(
                                normality_base_rate, auroc, n_aupr, ab_aupr))
                            monitor.record_matplot(auroc, gan_iter, 'abald_auroc')
                            monitor.record_matplot(n_aupr, gan_iter, 'abald_aupr+')
                            monitor.record_matplot(ab_aupr, gan_iter, 'abald_aupr-')

                    print


                if iteration > 0 and iteration % (sample_size*sample_interval) == 0:
                    if not arguments['--dont-save']:
                        # print("Saving {} samples...".format(sample_size))
                        print ("Saving samples to disc")
                        posterior_samples = []
                        for m in models: posterior_samples += m.posterior_samples
                        np_samples = _flatten_npyfy(posterior_samples)
                        np.save('./saves/{}/params_{:04d}'.format(exp_name, iteration//(sample_size*sample_interval)), np_samples)

                        total_num_samples_collected += np_samples.shape[0]
                        print("Total num samples collected: {}\n".format(total_num_samples_collected))
                    ###
                    check_point(model, optimizer, iteration, exp_name)
                    monitor.save_result_numpy(log_folder)

                ## apd
                if arguments['--apd'] and iteration > 0 and iteration % (apd_config['T_sgld']*sample_interval) == 0:

                        posterior_samples = []
                        for m in models: posterior_samples += m.posterior_samples[-apd_config['T_sgld']:]
                        np_samples = _flatten_npyfy(posterior_samples)
                        gan_inp_buffer += list(np_samples)
                        if len(gan_inp_buffer) > gan_bs:
                            for _ in range(apd_config['T_gan']):
                                idxs = np.arange(len(gan_inp_buffer))
                                np.random.shuffle(idxs)
                                obj_traingan.update_iter(gan_iter, np.array(gan_inp_buffer)[idxs[:gan_bs]])
                                gan_iter+=1
                            ## remove old MCMC samples
                            while len(gan_inp_buffer)>apd_config['apd_buffer_size']: del gan_inp_buffer[0]
                        ### infinite chains
                        if 'is_infinite_chains' in apd_config and apd_config['is_infinite_chains']:
                            sds = obj_traingan.get_samples(len(models))
                            for (idx, m) in enumerate(models):
                                m.load_state_dict(sds[idx])

                iteration = iteration + 1

    except (KeyboardInterrupt, ValueError):  # The ValueError should catch NaNs
        print('-' * 89)
        print('Exiting from training early!')


    ##################################
    ### Termination -- Run on test ###
    ##################################

    print('=' * 80)
    print('Test')
    print('=' * 80)

    num_test_runs = opt_config['num_test_runs']

    # # Prepare to store table results
    # cls_table = []

    # ad_headers = ['Dataset'] + ['ROC', 'PR(+)', 'PR(-)'] * 3
    # ad_table = []

    # table = [['Uniform', '96.99', '97.99', '94.71', '98.9', '99.15', '98.63', '98.97', '99.27', '98.52']]
    # print(tabulate.tabulate(table, headers, tablefmt='latex_booktabs'))



    ###########################
    ### Test Classification ###
    ###########################

    num_samples = opt_config['num_test_samples']

    if opt_config['name'] == 'NoisedSGD':
        posterior_flag = 1
    else:
        posterior_flag = 0

    # If using NSGD or SGLD, load several samples from the saves directory and evaluate the expectation of their predictions
    if opt_config['name'] == 'NoisedSGD':
        # posterior_flag = 1
        accuracy_list = []
        for i in range(num_test_runs):
            print("Test run {}".format(i))
            posterior_samples = utils.load_posterior_state_dicts(src_dir=exp_name, example_model=model, num_samples=num_samples)
            posterior_weights = [1 for _ in range(len(posterior_samples))]
            model.posterior_samples = posterior_samples  # Should change this structure
            model.posterior_weights = posterior_weights
            point_accuracy, point_loss, posterior_accuracy, posterior_loss = evaluate(model, testloader,posterior_flag, Loss, opt_config)
            print("Posterior acc: {}".format(posterior_accuracy))
            accuracy_list.append(posterior_accuracy)

        print("Sampling Test Results")
        print("---------------------")
        print("Test Accuracy: Mean {}, Std {}\n".format(np.mean(accuracy_list), np.std(accuracy_list)))

    #######################
    ### Test MC-Dropout ###
    #######################

    if arguments['--mc_dropout_passes'] and int(arguments['--mc_dropout_passes']) > 0:
        # Re-load model with best val accuracy
        best_model_state_dict = torch.load(os.path.join('saves', exp_name, 'best_mc_model.th'))
        model.load_state_dict(best_model_state_dict)

        mc_accuracy_list = []
        for i in range(num_test_runs):
            test_mc_accuracy, test_mc_loss = run_test_mc_dropout(model, testloader, arguments)
            mc_accuracy_list.append(test_mc_accuracy)

        print("MC-Dropout Test Results")
        print("-----------------------")
        print("Test Accuracy: Mean {}, Std {}\n".format(np.mean(mc_accuracy_list), np.std(mc_accuracy_list)))


    best_model_state_dict = torch.load(os.path.join('saves', exp_name, 'best_point_model.th'))
    model.load_state_dict(best_model_state_dict)

    point_accuracy, point_loss, posterior_accuracy, posterior_loss = evaluate(model, testloader, posterior_flag, Loss, opt_config)

    if posterior_accuracy:
        print("Point Acc: {:.4f} | Posterior Acc: {:.4f}".format(point_accuracy, posterior_accuracy))
    else:
        print("Point Acc: {:.4f}".format(point_accuracy))



    ##############################
    ### Test Anomaly detection ###
    ##############################

    anom_result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for scale in opt_config['test_ood_scales']:
        scale = int(scale)

        opt_config['ood_scale'] = scale
        ood_data = utils.load_ood_data(name_dataset, opt_config)

        print("OOD SCALE {}".format(scale))
        print("--------------------------")

        test_inputs_anomaly_detection = utils.get_anomaly_detection_test_inputs(testloader, opt_config, arguments)
        if arguments['--cuda']:
            test_inputs_anomaly_detection = test_inputs_anomaly_detection.cuda()
            for key in ood_data:
                ood_data[key] = ood_data[key].cuda()


        ############################################
        ### Test Deterministic Anomaly Detection ###
        ############################################

        # Load model with best val accuracy
        best_model_state_dict = torch.load(os.path.join('saves', exp_name, 'best_point_model.th'))
        model.load_state_dict(best_model_state_dict)

        for ood_dataset_name in opt_config['ood_datasets']:
            print("OOD Dataset: {}".format(ood_dataset_name))

            cur_ood_data = ood_data[ood_dataset_name]

            for func_name in opt_config['ood_acq_funcs']:

                if func_name != 'f_bald':
                    # Non-bayesian
                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(
                        test_inputs_anomaly_detection, cur_ood_data, utils.sm_given_np_data, {'model': model}, f_acq=func_name)
                    print(
                        "({} Non-Bayesian) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(
                            func_name, normality_base_rate, auroc, n_aupr, ab_aupr))

                    anom_result_dict[scale][ood_dataset_name][func_name] = [auroc, n_aupr, ab_aupr]

            # anomaly_detection_monitor.record_matplot([auroc, n_aupr, ab_aupr], iteration, 'point_estimation')

        #################################################
        ### Test Bayesian (Sampled) Anomaly Detection ###
        #################################################
        if posterior_flag == 1:
            print

            for ood_dataset_name in opt_config['ood_datasets']:
                print("OOD Dataset: {}".format(ood_dataset_name))

                cur_ood_data = ood_data[ood_dataset_name]

                for func_name in opt_config['ood_acq_funcs']:
                    normality_base_rate_list = []
                    auroc_list = []
                    n_aupr_list = []
                    ab_aupr_list = []

                    for i in range(num_test_runs):
                        posterior_samples = utils.load_posterior_state_dicts(src_dir=exp_name, example_model=model, num_samples=num_samples)
                        posterior_weights = [1 for _ in range(len(posterior_samples))]
                        model.posterior_samples = posterior_samples  # Should change this structure
                        model.posterior_weights = posterior_weights

                        # normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection, cur_ood_data, utils.posterior_expectation,
                        #                                                                                        # {'model': model})
                        #                                                                                        {'model': model, 'use_mini_batch': opt_config['batch_size']})
                        if func_name == 'f_bald':
                            normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                   cur_ood_data,
                                                                                                                   utils.posterior_expectation,
                                                                                                                   {'model': model, 'keep_samples': True, 'use_mini_batch': opt_config['batch_size']},
                                                                                                                   f_acq='f_bald')
                        else:
                            normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                   cur_ood_data,
                                                                                                                   utils.posterior_expectation,
                                                                                                                   {'model': model, 'keep_samples': False, 'use_mini_batch': opt_config['batch_size']},
                                                                                                                   f_acq=func_name)

                        normality_base_rate_list.append(normality_base_rate)
                        auroc_list.append(auroc)
                        n_aupr_list.append(n_aupr)
                        ab_aupr_list.append(ab_aupr)

                    print("({} Bayesian) Anomaly Detection Results: \nBase Rate: {:.2f}/{:.3f}, AUROC: {:.2f}/{:.3f}, AUPR+: {:.2f}/{:.3f}, AUPR-: {:.2f}/{:.3f}".format(
                          func_name,
                          np.mean(normality_base_rate_list), np.std(normality_base_rate_list),
                          np.mean(auroc_list), np.std(auroc_list),
                          np.mean(n_aupr_list), np.std(n_aupr_list),
                          np.mean(ab_aupr_list), np.std(ab_aupr_list)))

                    anom_result_dict[scale][ood_dataset_name][func_name] = [(np.mean(auroc_list), np.std(auroc_list)),
                                                                            (np.mean(n_aupr_list), np.std(n_aupr_list)),
                                                                            (np.mean(ab_aupr_list), np.std(ab_aupr_list))]

                # anomaly_detection_monitor.record_matplot([auroc, n_aupr, ab_aupr], iteration, 'bayesian')


        #########################################
        ### Test MC-Dropout Anomaly Detection ###
        #########################################

        if arguments['--mc_dropout_passes'] and int(arguments['--mc_dropout_passes']) > 0:
            print

            # Re-load model with best val accuracy
            best_model_state_dict = torch.load(os.path.join('saves', exp_name, 'best_mc_model.th'))
            model.load_state_dict(best_model_state_dict)

            for ood_dataset_name in opt_config['ood_datasets']:
                print("OOD Dataset: {}".format(ood_dataset_name))
                cur_ood_data = ood_data[ood_dataset_name]

                for func_name in opt_config['ood_acq_funcs']:
                    normality_base_rate_list = []
                    auroc_list = []
                    n_aupr_list = []
                    ab_aupr_list = []

                    for i in range(num_test_runs):
                        if func_name != 'f_bald':
                            normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection,
                                                                                                                   cur_ood_data,
                                                                                                                   utils.mc_dropout_expectation,
                                                                                                                   {'model': model, 'passes': arguments['--mc_dropout_passes'], 'keep_samples': False},
                                                                                                                   f_acq=func_name)
                        # normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs_anomaly_detection, cur_ood_data, utils.mc_dropout_expectation,
                        #                                                                                        {'model': model, 'passes': arguments['--mc_dropout_passes']})
                        normality_base_rate_list.append(normality_base_rate)
                        auroc_list.append(auroc)
                        n_aupr_list.append(n_aupr)
                        ab_aupr_list.append(ab_aupr)

                    # print("({}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(func_name.upper(), normality_base_rate, auroc, n_aupr, ab_aupr))
                    print("({} MC-Dropout) Anomaly Detection Results: \nBase Rate: {:.2f}/{:.3f}, AUROC: {:.2f}/{:.3f}, AUPR+: {:.2f}/{:.3f}, AUPR-: {:.2f}/{:.3f}".format(
                          func_name,
                          np.mean(normality_base_rate_list), np.std(normality_base_rate_list),
                          np.mean(auroc_list), np.std(auroc_list),
                          np.mean(n_aupr_list), np.std(n_aupr_list),
                          np.mean(ab_aupr_list), np.std(ab_aupr_list)))

                    anom_result_dict[scale][ood_dataset_name][func_name] = [(np.mean(auroc_list), np.std(auroc_list)),
                                                                            (np.mean(n_aupr_list), np.std(n_aupr_list)),
                                                                            (np.mean(ab_aupr_list), np.std(ab_aupr_list))]

    for scale in opt_config['test_ood_scales']:
        for ood_dataset_name in opt_config['ood_datasets']:
            anom_result_dict[scale][ood_dataset_name] = dict(anom_result_dict[scale][ood_dataset_name])
        anom_result_dict[scale] = dict(anom_result_dict[scale])

    anom_result_dict = dict(anom_result_dict)


    with open(os.path.join('saves', exp_name, 'anom_res_s:{}'.format(opt_config['num_test_samples'])), 'wb') as f:
        pkl.dump(anom_result_dict, f)

    print('-' * 89)
    print('Experiment directory: {}'.format(os.path.join('saves', exp_name)))
    print('-' * 89)

    # loss_monitor.save_result_numpy(log_folder)
    # mean, std = anomaly_detection_monitor.statistics_result('anomaly_detection', 100)
    # print("Mean: {}".format(mean))
    # print("Std: {}".format(std))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt...")
    print(arguments)
    print("............\n")

    main(arguments)
