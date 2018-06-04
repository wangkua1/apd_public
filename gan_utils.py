import os, errno
import pdb
import yaml
import numpy as np
import tflib as lib
"""
torch...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

"""
our models/utilities
"""
import utils
# from utils import inference_accuracy, posterior_expectation, show_ood_detection_results_softmax
from model.fc import *
from model.cnn import *
from opt.loss import *

Loss = CE()
tft = lambda x: torch.FloatTensor(x)
tfv = lambda x: Variable(tft(x))

"""
GAN related
"""

def inf_train_gen(train_gen):
    while True:
        for images in train_gen():
            # print ('db...')
            # print (images.shape)
            yield images


def data_generator(src_dir, batch_size, N_TRAIN, sampling_type='random'):

    selected = utils.load_posterior_samples(src_dir, N_TRAIN, sampling_type=sampling_type)
    data = np.vstack(selected)

    # data = np.concatenate([np.load(os.path.join('saves', src_dir, f_datum)) for f_datum in f_data], 0)
    # data = data[:N_TRAIN]

    print("Num parameters: {}".format(data.shape[1]))

    def get_epoch():
        # rng_state = np.random.get_state()
        np.random.shuffle(data)
        # np.random.set_state(rng_state)
        for i in range(len(data) // batch_size):
            yield data[i*batch_size:(i+1)*batch_size]

    return get_epoch


def model_load_configuration(arguments):
    """
    infer/load target model by parsing src_dir
    """
    src_dir = arguments['<src_dir>']
    # f_model_config = 'model/config/' + src_dir.split('.')[1].split('-X-')[0] + '.yaml'
    f_model_config = 'model/config/' + src_dir.split('-X-')[0] + '.yaml'
    model_config = yaml.load(open(f_model_config, 'rb'))
    model = eval(model_config['name'])(**model_config['kwargs'])
    return model


def opt_load_configuration(f_opt_config, default_config):
    if not default_config:
        default_config = {
            'n_train': 100000,
            'task': 'mnist',
            'batch_size': 100,  # Batch size  (what's the diff between batch size and sample_size??)
            'validation_interval': 2000,
            'ood_scale': 5,
            'gan_dim': 100,  # The size of the hidden state in the generator and discriminator
            'zdim': 32,
            'lambda': 10,  # Gradient penalty lambda hyperparameter
            'critic_iters': 5,  # How many critic iterations per generator iteration
            'iters': 200000,  # How many generator iterations to train for
            # 'sampling_type': 'first',
            'sampling_type': 'random'
        }

    opt_config = yaml.load(open(f_opt_config, 'rb'))

    # Fill in default configuration for keys that are not overwritten by the config file
    for key in default_config:
        if key not in opt_config:
            opt_config[key] = default_config[key]

    # Fill in default n_anom value, if not overwritten by the config file
    if 'n_anom' not in opt_config:
        if name_task.split('-')[0] in ['mnist', 'fashion']:
            opt_config['n_anom'] = 2000
        elif name_task.split('-')[0] == 'mnist':
            opt_config['n_anom'] = 300

    return opt_config


"""
BNN (i.e. samples) related
"""
def evaluate(model, testloader, posterior_samples, posterior_weights, posterior_flag, Loss, opt_config, arguments):
    model.eval()

    point_accuracy = []
    point_loss = []
    posterior_accuracy = []
    posterior_loss = []

    for i, data in enumerate(testloader, 0):

        # Load data
        # data for inference
        test_inputs, test_labels = data
        if arguments['--cuda']:
            test_inputs = test_inputs.cuda()
            test_labels = test_labels.cuda()
        test_inputs, test_labels = Variable(test_inputs, volatile=True), Variable(test_labels, volatile=True)

        # Prediction
        # Point Prediction
        point_outputs = model.forward(test_inputs)
        point_loss_batch = F.cross_entropy(point_outputs.cpu(), test_labels.cpu())
        point_loss.append(point_loss_batch.data[0])

        # point_predictions = Loss.inference_prediction(point_outputs)
        prob_inputs = F.softmax(point_outputs)
        point_predictions = prob_inputs.data.cpu().numpy().argmax(1)

        point_accuracy_batch = utils.inference_accuracy(point_predictions, test_labels)
        point_accuracy.append(point_accuracy_batch)

        # Bayesian Prediction
        if posterior_flag:
            posterior_outputs = utils.posterior_expectation(model, test_inputs)
            # posterior_loss_batch = Loss.nll(torch.log(posterior_outputs), test_labels)
            posterior_loss_batch = F.nll_loss(torch.log(posterior_outputs.cpu()), test_labels.cpu())
            posterior_loss.append(posterior_loss_batch.data[0])

            # posterior_predictions = Loss.inference_prediction(posterior_outputs)
            prob_inputs = F.softmax(posterior_outputs)
            posterior_predictions = prob_inputs.data.cpu().numpy().argmax(1)

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


def get_anomaly_detection_test_inputs(testloader, opt_config):
    n_anom = opt_config['n_anom']
    num_batch_anomaly_detection = (n_anom // opt_config['test_input_batch_size']) + 1  # Be careful about what this batch size means!
    ood_test_inputs = []

    dataiter = iter(testloader)

    for _ in range(num_batch_anomaly_detection):
        test_inputs, _ = dataiter.next()
        ood_test_inputs.append(test_inputs)

    ood_test_inputs = torch.cat(ood_test_inputs, 0)[:n_anom]
    ood_test_inputs = Variable(ood_test_inputs, volatile=True)
    return ood_test_inputs


class EvalToy2d(object):
    """docstring for EvalToy2d"""
    def __init__(self, model):
        super(EvalToy2d, self).__init__()
        ## Data
        size = 10
        m1 = [-2,-2]
        m2 = [2,2]
        cov = np.eye(2) * .3
        x1 = np.random.multivariate_normal(m1, cov,size=size)
        x2 = np.random.multivariate_normal(m2, cov,size=size)
        X = np.vstack([x1,x2])
        Y = np.zeros((size*2, 2))
        Y[:size,0] = 1
        Y[size:,1] = 1

        linspace = np.arange(-5,5,0.1)
        ###
        self.model = model
        self.test_points = np.array(list(itertools.product(linspace, linspace)))
        self.X = tfv(X)
        self.Y = tft(Y)

    def _plot(self,test_points, py1_xs ):
        im = np.zeros((99,99))
        for (py1_x, tp) in zip(py1_xs, test_points):
            row, col = int(tp[0]*10+50), int(tp[1]*10+50)
            im[row, col] = py1_x
        return im

    def toy2d_validate(self,posterior_samples, idx):
        model = self.model
        X = self.X
        Y = self.Y
        test_points = self.test_points

        posterior_weights = [1 for _ in range(len(posterior_samples))]
        def _validate_batch_bayes(posterior_samples,posterior_weights, X_val_batch):
            model.eval()
            acc_proba = None
            for sample_idx in range(len(posterior_samples)):
                p_sample = posterior_samples[sample_idx]
                model.load_state_dict(p_sample)
                _, proba = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor), ret_proba=True)
                if acc_proba is None:
                    acc_proba = posterior_weights[sample_idx] * proba
                else:
                    acc_proba += posterior_weights[sample_idx] * proba
            model.train()
            acc_proba /= sum(posterior_weights)
            return acc_proba[:,0]
        bayes_probas = _validate_batch_bayes(posterior_samples,posterior_weights, test_points)
        npX = X.data.numpy()
        plt.scatter(npX[:size,0]*10+50,npX[:size,1]*10+50,c='k')
        plt.scatter(npX[size:,0]*10+50,npX[size:,1]*10+50,c='w')

        plt.imshow(self._plot(self.test_points, bayes_probas), cmap=plt.cm.rainbow,interpolation='bicubic')
        plt.savefig('bayes_probas_%g.png'%idx)

class EvalMNIST(object):
    """docstring for EvalMNIST"""
    def __init__(self, model, opt_config, cuda, log_dir=None):
        super(EvalMNIST, self).__init__()
        self.model = model
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt_config['test_input_batch_size'], shuffle=False, num_workers=2)
        dataiter = iter(testloader)
        test_inputs, test_labels = dataiter.next()
        if cuda:
            test_inputs = test_inputs.cuda()
            test_labels = test_labels.cuda()

        test_inputs, test_labels = Variable(test_inputs, volatile=True), Variable(test_labels, volatile=True)
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            log_dir = 'gan_mnist_logs'
            try:
                os.makedirs(log_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.accuracy_log_file = open(os.path.join(log_dir, 'acc_log.txt'), 'w')
        self.anom_log_file = open(os.path.join(log_dir, 'anom_log.txt'), 'w')
        self.real_acc_log_file = open(os.path.join(log_dir, 'real_acc_log.txt'), 'w')
        self.real_anom_log_file = open(os.path.join(log_dir, 'real_anom_log.txt'), 'w')

        #########################
        ### Anomaly Detection ###
        #########################
        ood_data = utils.load_ood_data('mnist', opt_config)
        ood_test_inputs = get_anomaly_detection_test_inputs(testloader, opt_config)

        if cuda:
            ood_test_inputs = ood_test_inputs.cuda()
            for key in ood_data:
                ood_data[key] = ood_data[key].cuda()

        ##
        self.model = model
        self.testloader = testloader
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.ood_data = ood_data
        self.ood_test_inputs = ood_test_inputs
        self.opt_config = opt_config

    def mnist_validate(self, posterior_samples, iteration, log=None):
        print("MNIST VALIDATE")
        model  = self.model
        test_inputs = self.test_inputs
        test_labels = self.test_labels

        model.posterior_samples = posterior_samples
        model.posterior_weights = [1 for _ in range(len(posterior_samples))]
        posterior_outputs = utils.posterior_expectation(model, test_inputs)
        # posterior_loss = F.cross_entropy(posterior_outputs.cpu(), test_labels)

        posterior_loss = F.nll_loss(torch.log(posterior_outputs.cpu()), test_labels.cpu())
        utils.check_nan(posterior_loss, check_big=True, message="")

        posterior_predictions = Loss.inference_prediction(posterior_outputs)  # Checked
        posterior_accuracy = utils.inference_accuracy(posterior_predictions, test_labels)  # Checked
        print("Classification acc: {:.4f}".format(posterior_accuracy))
        print("loss: {:.4f}".format(posterior_loss.data[0]))
        lib.plot.plot('classification acc', posterior_accuracy)

        if log == 'fake':
            # Write to log file
            self.accuracy_log_file.write('{} {}\n'.format(iteration, posterior_accuracy))
            self.accuracy_log_file.flush()
        elif log == 'real':
            self.real_acc_log_file.write('{} {}\n'.format(iteration, posterior_accuracy))
            self.real_acc_log_file.flush()

    def mnist_ood(self, posterior_samples, iteration, log=None):
        print("MNIST OOD")
        model  = self.model
        ood_data = self.ood_data
        ood_test_inputs = self.ood_test_inputs
        opt_config = self.opt_config

        model.posterior_samples = posterior_samples
        model.posterior_weights = [1 for _ in range(len(posterior_samples))]

        # Bayesian
        for ood_dataset_name in opt_config['ood_datasets']:
            print("OOD Dataset: {}".format(ood_dataset_name))
            cur_ood_data = ood_data[ood_dataset_name]
            for func_name in opt_config['ood_acq_funcs']:
                if func_name == 'f_bald':
                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(ood_test_inputs,
                                                                                                           cur_ood_data,
                                                                                                           utils.posterior_expectation,
                                                                                                           {'model': model, 'keep_samples': True, 'use_mini_batch': opt_config['batch_size']},
                                                                                                           f_acq='f_bald')
                else:
                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(ood_test_inputs,
                                                                                                           cur_ood_data,
                                                                                                           utils.posterior_expectation,
                                                                                                           {'model': model, 'keep_samples': False, 'use_mini_batch': opt_config['batch_size']},
                                                                                                           f_acq=func_name)
                print("({}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(func_name.upper(), normality_base_rate, auroc, n_aupr, ab_aupr))

                if log == 'fake':
                    # Write to log file
                    self.anom_log_file.write('{} {} {} {} {} {} {}\n'.format(iteration,
                                                                             ood_dataset_name,
                                                                             func_name,
                                                                             normality_base_rate,
                                                                             auroc,
                                                                             n_aupr,
                                                                             ab_aupr))
                    self.anom_log_file.flush()
                elif log == 'real':
                    # Write to log file
                    self.real_anom_log_file.write('{} {} {} {} {} {} {}\n'.format(iteration,
                                                                                  ood_dataset_name,
                                                                                  func_name,
                                                                                  normality_base_rate,
                                                                                  auroc,
                                                                                  n_aupr,
                                                                                  ab_aupr))
                    self.real_anom_log_file.flush()


class EvalCIFAR(object):
    """docstring for EvalMNIST"""
    def __init__(self, model, opt_config, cuda, log_dir=None):
        super(EvalCIFAR, self).__init__()
        self.model = model
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        first_5_class_idxs = [i for i in range(len(testset.test_labels)) if testset.test_labels[i] in [0,1,2,3,4]]
        testset.test_data = np.stack([testset.test_data[i, :, :, :] for i in first_5_class_idxs])
        testset.test_labels = np.stack([testset.test_labels[i] for i in first_5_class_idxs])

        testloader = torch.utils.data.DataLoader(testset, batch_size=opt_config['test_input_batch_size'], shuffle=False, num_workers=2)
        dataiter = iter(testloader)
        test_inputs, test_labels = dataiter.next()
        if cuda:
            test_inputs = test_inputs.cuda()
            test_labels = test_labels.cuda()

        test_inputs, test_labels = Variable(test_inputs, volatile=True), Variable(test_labels, volatile=True)

        self.log_dir = log_dir
        self.accuracy_log_file = open(os.path.join(log_dir, 'acc_log.txt'), 'w')
        self.anom_log_file = open(os.path.join(log_dir, 'anom_log.txt'), 'w')
        self.real_acc_log_file = open(os.path.join(log_dir, 'real_acc_log.txt'), 'w')
        self.real_anom_log_file = open(os.path.join(log_dir, 'real_anom_log.txt'), 'w')

        #########################
        ### Anomaly Detection ###
        #########################
        ood_data = utils.load_ood_data('cifar5', opt_config)
        ood_test_inputs = get_anomaly_detection_test_inputs(testloader, opt_config)

        if cuda:
            ood_test_inputs = ood_test_inputs.cuda()
            for key in ood_data:
                ood_data[key] = ood_data[key].cuda()

        ##
        self.model = model
        self.testloader = testloader
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.ood_data = ood_data
        self.ood_test_inputs = ood_test_inputs
        self.opt_config = opt_config

    def cifar_validate(self, posterior_samples, iteration, log=None):
        model  = self.model
        model.eval()  # To turn off dropout that may be present by default
        test_inputs = self.test_inputs
        test_labels = self.test_labels

        model.posterior_samples = posterior_samples
        model.posterior_weights = [1 for _ in range(len(posterior_samples))]
        posterior_outputs = utils.posterior_expectation(model, test_inputs)

        posterior_loss = F.cross_entropy(posterior_outputs.cpu(), test_labels.cpu())

        # posterior_loss = F.nll_loss(torch.log(posterior_outputs.cpu()), test_labels.cpu())

        posterior_predictions = Loss.inference_prediction(posterior_outputs)  # Checked
        posterior_accuracy = utils.inference_accuracy(posterior_predictions, test_labels)  # Checked
        print("Classification acc: {:.4f}".format(posterior_accuracy))
        print("loss: {:.4f}".format(posterior_loss.data[0]))
        lib.plot.plot('classification acc', posterior_accuracy)

        if log == 'fake':
            # Write to log file
            self.accuracy_log_file.write('{} {}\n'.format(iteration, posterior_accuracy))
            self.accuracy_log_file.flush()
        elif log == 'real':
            self.real_acc_log_file.write('{} {}\n'.format(iteration, posterior_accuracy))
            self.real_acc_log_file.flush()

        model.train()  # To re-enable dropout

    def cifar_ood(self, posterior_samples, iteration, log=None):
        model = self.model
        model.eval()  # Disable dropout
        ood_data = self.ood_data
        ood_test_inputs = self.ood_test_inputs
        opt_config = self.opt_config

        model.posterior_samples = posterior_samples
        model.posterior_weights = [1 for _ in range(len(posterior_samples))]

        # Bayesian
        for ood_dataset_name in opt_config['ood_datasets']:
            print("OOD Dataset: {}".format(ood_dataset_name))
            cur_ood_data = ood_data[ood_dataset_name]
            for func_name in opt_config['ood_acq_funcs']:
                if func_name == 'f_bald':
                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(ood_test_inputs,
                                                                                                           cur_ood_data,
                                                                                                           utils.posterior_expectation,
                                                                                                           {'model': model, 'keep_samples': True, 'use_mini_batch': opt_config['batch_size']},
                                                                                                           f_acq='f_bald')
                else:
                    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(ood_test_inputs,
                                                                                                           cur_ood_data,
                                                                                                           utils.posterior_expectation,
                                                                                                           {'model': model, 'keep_samples': False, 'use_mini_batch': opt_config['batch_size']},
                                                                                                           f_acq=func_name)
                print("({}) Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(func_name.upper(), normality_base_rate, auroc, n_aupr, ab_aupr))

                if log == 'fake':
                    # Write to log file
                    self.anom_log_file.write('{} {} {} {} {} {} {}\n'.format(iteration,
                                                                             ood_dataset_name,
                                                                             func_name,
                                                                             normality_base_rate,
                                                                             auroc,
                                                                             n_aupr,
                                                                             ab_aupr))
                    self.anom_log_file.flush()
                elif log == 'real':
                    # Write to log file
                    self.real_anom_log_file.write('{} {} {} {} {} {} {}\n'.format(iteration,
                                                                                  ood_dataset_name,
                                                                                  func_name,
                                                                                  normality_base_rate,
                                                                                  auroc,
                                                                                  n_aupr,
                                                                                  ab_aupr))
                    self.real_anom_log_file.flush()

        model.train()  # Re-enable dropout
