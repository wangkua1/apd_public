import os
import sys
import pdb
import math
import copy
import pickle

from collections import OrderedDict

import numpy as np
import sklearn.metrics as sk

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms


def sm_given_np_data(inputs, model):
    model.eval()
    sm_oos = F.softmax(model.forward(inputs))
    return sm_oos


def inference_accuracy(prediction, labels):
    accuracy = (prediction == labels.data.cpu().numpy()).mean().astype(float)
    return accuracy


def load_posterior_state_dicts(src_dir, example_model, num_samples):
    samples = load_posterior_samples(src_dir, num_samples)
    return prepare_torch_dicts(samples, example_model)


def load_posterior_samples(src_dir, num_samples, sampling_type='random'):
    """Choices for sampling type: random, first, last
    """
    data_files = [s for s in os.listdir(os.path.join('saves', src_dir)) if s.split('.')[-1] == 'npy']

    if sampling_type == 'first':
        data_files = sorted(data_files)  # Sort based on sample number in filename

        samples = []
        for f in data_files:
            if len(samples) <= num_samples:
                tmp_data = np.load(os.path.join('saves', src_dir, f))
                samples.append(tmp_data)

    elif sampling_type == 'last':
        data_files = sorted(data_files, reverse=True)  # Reverse sort based on sample number in filename

        samples = []
        for f in data_files:
            if len(samples) <= num_samples:
                tmp_data = np.load(os.path.join('saves', src_dir, f))
                samples.append(tmp_data)

    elif sampling_type == 'random':
        num_from_each_file = num_samples // len(data_files) + 1

        samples = []

        np.random.shuffle(data_files)

        for f in data_files:
            if len(samples) <= num_samples:
                tmp_data = np.load(os.path.join('saves', src_dir, f))
                num_items = tmp_data.shape[0]
                samples += [tmp_data[i] for i in np.random.randint(0, num_items-1, num_from_each_file)]

    samples = samples[:num_samples]
    return samples


# def load_posterior_samples(src_dir, num_samples):
#     data_files = [s for s in os.listdir(os.path.join('saves', src_dir)) if s.split('.')[-1] == 'npy']
#     num_from_each_file = num_samples // len(data_files) + 1

#     samples = []

#     np.random.shuffle(data_files)

#     for f in data_files:
#         if len(samples) <= num_samples:
#             tmp_data = np.load(os.path.join('saves', src_dir, f))
#             num_items = tmp_data.shape[0]
#             samples += [tmp_data[i] for i in np.random.randint(0, num_items-1, num_from_each_file)]

#     samples = samples[:num_samples]
#     return samples


def prepare_torch_dicts(params, example_model):
    ds = example_model.state_dict()
    ret = []
    for p in params:
        curr_idx = 0
        currls = []
        for d in ds:
            curr = OrderedDict()
            for (k, elem_d) in d.items():
                size = np.array(elem_d.size())
                curr[k] = torch.FloatTensor(p[curr_idx:curr_idx+np.prod(size)].reshape(size))
                curr_idx += np.prod(size)
            currls.append(curr)
        ret.append(currls)
    return ret


def mc_dropout_expectation(model, inputs, keep_samples=False, passes=50):
    model.train()
    output_probs_list = []
    for i in range(int(passes)):

        outputs = model.forward(inputs)
        outputs = outputs.cpu()
        output_probs = F.softmax(outputs)
        output_probs_list.append(output_probs)

    mean_probs = torch.mean(torch.stack(output_probs_list, dim=1), dim=1)
    if not keep_samples:
        return mean_probs #(n,d)
    else:
        return mean_probs, output_probs_list


def get_anomaly_detection_test_inputs(testloader, opt_config, arguments):
    n_anom = opt_config['n_anom']
    if 'batcher_kwargs' in opt_config:
        num_batch_anomaly_detection = (n_anom // opt_config['batcher_kwargs']['batch_size']) + 1
    else:
        num_batch_anomaly_detection = (n_anom // opt_config['batch_size']) + 1
    test_inputs_anomaly_detection = []
    for i, data in enumerate(testloader, 0):
        test_inputs, test_labels = data
        test_inputs, test_labels = cuda((test_inputs, test_labels), arguments)
        test_inputs = Variable(test_inputs, volatile=True)
        if i <= num_batch_anomaly_detection:
            test_inputs_anomaly_detection.append(test_inputs)

    test_inputs_anomaly_detection = torch.cat(test_inputs_anomaly_detection, 0)[:n_anom]
    return test_inputs_anomaly_detection


def posterior_expectation(model, inputs, keep_samples=False, use_mini_batch=None, sample_size=None):
    posterior_samples = model.posterior_samples
    posterior_weights = model.posterior_weights
    if sample_size is not None:
        idxs = np.arange(len(posterior_samples)).astype('int32')
        np.random.shuffle(idxs)
        idxs  = idxs[:sample_size]
        posterior_samples = copy.deepcopy(list(np.array(posterior_samples)[idxs]))
        posterior_weights = copy.deepcopy(list(np.array(posterior_weights)[idxs]))

    num_posterior_samples = len(posterior_samples)
    outputs_weighted_sum = 0
    model.eval()
    sample_sms = []
    ### cache the current model param, restore later
    cached_params = copy.deepcopy(model.state_dict())
    for sample_idx in range(num_posterior_samples):

        model.load_state_dict(posterior_samples[sample_idx])

        if use_mini_batch is not None:
            outputs_samples = []
            N = inputs.size()[0]
            num_batches = int(math.ceil(N / float(use_mini_batch)))
            for batch_idx in range(num_batches):
                start_idx = batch_idx * use_mini_batch
                end_idx = min(start_idx + use_mini_batch, N)
                batch_output = model.forward(inputs[start_idx:end_idx])
                # outputs_samples.append(batch_output.cpu())
                outputs_samples.append(batch_output)
            outputs_samples = torch.cat(outputs_samples,0)
        else:
            outputs_samples = model.forward(inputs)

        outputs_prob = F.softmax(outputs_samples).cpu() ## otherwise run out of gpu memory
        sample_sms.append(outputs_prob)
        # outputs_sample --> softmax --> prob
        outputs_weighted_sum = outputs_weighted_sum + outputs_prob * float(posterior_weights[sample_idx])
    model.train()
    outputs_expectation = outputs_weighted_sum / float(sum(posterior_weights))

    # outputs_expectation = outputs_expectation.cuda() ## put it back on gpu
    # restore
    model.load_state_dict(cached_params)
    if not keep_samples:
        # return outputs_expectation.cuda() #(n,d)
        return outputs_expectation #(n,d)
    else:
        return outputs_expectation, sample_sms


def posterior_uncertainty(model, inputs):
    '''
    Uncertainty estimate given in https://arxiv.org/pdf/1703.00410.pdf
    Equation (6).
    '''
    posterior_samples = model.posterior_samples
    posterior_weights = model.posterior_weights
    num_posterior_samples = len(posterior_samples)
    outputs_weighted_sum = 0
    outputs_squared_sum = 0
    model.eval()
    for sample_idx in range(num_posterior_samples):

        # New
        model.load_state_dict(posterior_samples[sample_idx])

        outputs_samples = model.forward(inputs)
        outputs_prob = F.softmax(outputs_samples)

        # outputs_sample --> softmax --> prob
        outputs_weighted_sum = outputs_weighted_sum + outputs_prob*posterior_weights[sample_idx]
        outputs_squared_sum = outputs_squared_sum + (outputs_prob*outputs_prob).sum(1) * posterior_weights[sample_idx]
    model.train()
    weighted_sum = sum(posterior_weights)
    outputs_expectation = outputs_weighted_sum / weighted_sum
    return outputs_squared_sum / weighted_sum - (outputs_expectation * outputs_expectation).sum(1)


def mc_dropout_uncertainty(model, inputs, passes=50):
    '''
    Uncertainty estimate given in https://arxiv.org/pdf/1703.00410.pdf
    Equation (6).
    '''
    model.train()
    outputs_sum = 0
    outputs_squared_sum = 0
    passes = int(passes)
    for i in range(passes):
        outputs = model.forward(inputs)
        outputs_prob = F.softmax(outputs)

        outputs_sum = outputs_sum + outputs_prob
        outputs_squared_sum = outputs_squared_sum + (outputs_prob*outputs_prob).sum(1)
    outputs_expectation = outputs_sum / passes
    return outputs_squared_sum / passes - (outputs_expectation * outputs_expectation).sum(1)


def _cpu_data(t):
    if isinstance(t, np.zeros(1).__class__):
        return t
    else:
        return t.data.cpu().numpy()


def f_uncert_x(inp):
    '''
    Uncertainty estimate given in https://arxiv.org/pdf/1703.00410.pdf
    Equation (6).
    '''
    sm, sample_sms = inp
    sm = _cpu_data(sm)
    sample_sms = np.array([_cpu_data(s) for s in sample_sms])

    squared_sum = (sample_sms * sample_sms).sum(-1).mean(0)
    return squared_sum - (sm * sm).sum(-1)


def f_bald(inp):
    sm, sample_sms = inp
    sm = _cpu_data(sm)
    # 1st term
    H = f_entropy(sm)
    # 2nd term
    v = np.array([f_entropy(_cpu_data(sm)) for sm in sample_sms]).mean(0)
    # bald
    bald = H-v
    return bald


def f_entropy(sm, smooth=0.000001):
    sm = _cpu_data(sm)

    sm = (sm+smooth)/1+smooth*sm.shape[-1]
    return -np.sum(sm * np.log(sm), -1)


def f_maxy(sm):
    sm = _cpu_data(sm)
    return -sm.max(-1)


def f_identity(sm):
    return _cpu_data(sm)


def show_ood_detection_results_softmax(in_examples, out_examples, f_pred, f_kwargs, f_acq='f_maxy'):

    f_acq = eval(f_acq)
    s_p_oos = f_acq(f_pred(inputs=out_examples,**f_kwargs))[:,None]
    s_p = f_acq(f_pred(inputs=in_examples,**f_kwargs))[:,None]

    normality_base_rate = round(100*in_examples.size()[0]/(
                out_examples.size()[0] + in_examples.size()[0]),2)
    # Prediction Prob: Normality Detection
    safe, risky = s_p, s_p_oos
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[safe.shape[0]:] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    n_aupr = round(100*sk.average_precision_score(labels, examples), 2)
    auroc = round(100*sk.roc_auc_score(labels, examples), 2)
    # Abnormality Detection
    safe, risky = -s_p, -s_p_oos
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    ab_aupr = round(100*sk.average_precision_score(labels, examples), 2)
    return normality_base_rate, auroc, n_aupr, ab_aupr


def load_ood_dataset(name_dataset, ood_dataset_name, opt_config):
    if name_dataset in ['mnist', 'fashion']:
        if ood_dataset_name == 'notMNIST':
            # N_ANOM = 2000
            pickle_file = './data/notMNIST.pickle'
            with open(pickle_file, 'rb') as f:

                try:
                    save = pickle.load(f, encoding='latin1')
                except TypeError:
                    save = pickle.load(f)

                ood_data = save['train_dataset'][:,None] * opt_config['ood_scale']  # (20000, 1, 28, 28)
                del save
            return ood_data
        elif ood_dataset_name == 'omniglot':  # Taken from https://github.com/hendrycks/error-detection
            import scipy.io as sio
            import scipy.misc as scimisc
            # other alphabets have characters which overlap
            safe_list = [0,2,5,6,8,12,13,14,15,16,17,18,19,21,26]
            m = sio.loadmat("./data/data_background.mat")

            squished_set = []
            for safe_number in safe_list:
                for alphabet in m['images'][safe_number]:
                    for letters in alphabet:
                        for letter in letters:
                            for example in letter:
                                # squished_set.append(scimisc.imresize(1 - example[0], (28,28)).reshape(1, 28*28))
                                squished_set.append(scimisc.imresize(1 - example[0], (28,28)))

            omniglot_images = np.stack(squished_set) * opt_config['ood_scale']
            ood_data = np.expand_dims(omniglot_images, axis=1)
            return ood_data.astype('float32')
        elif ood_dataset_name == 'cifar10bw':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            cifar10_batch_size = 10
            cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            cifar10_testloader = torch.utils.data.dataloader.DataLoader(cifar10_testset, batch_size=cifar10_batch_size, shuffle=False)
            cifar10_testiter = iter(cifar10_testloader)

            ood_data_list = []

            while True:
                try:
                    cifar10_images, _ = cifar10_testiter.next()
                    ood_data_list.append(cifar10_images)
                except StopIteration:
                    break

            ood_data = torch.cat(ood_data_list, 0)
            return ood_data.numpy() * opt_config['ood_scale']  # For consistency, all parts of this function return numpy arrays  (10000, 1, 28, 28)

        elif ood_dataset_name == 'gaussian':
            return np.random.normal(size=(opt_config['n_anom'], 1, 28, 28)) * opt_config['ood_scale']
        elif ood_dataset_name == 'uniform':
            return np.random.uniform(size=(opt_config['n_anom'], 1, 28, 28)) * opt_config['ood_scale']

    elif name_dataset == 'cifar5':
        if ood_dataset_name == 'cifar5_other':
            # 5 of the CIFAR-10 classes: dog, frog, horse, ship, truck
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            cifar5_batch_size = 10

            # Test set
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            last_5_class_idxs = [i for i in range(len(testset.test_labels)) if testset.test_labels[i] in [5,6,7,8,9]]
            testset.test_data = np.stack([testset.test_data[i, :, :, :] for i in last_5_class_idxs])
            testset.test_labels = np.stack([testset.test_labels[i] for i in last_5_class_idxs])
            cifar5_testloader = torch.utils.data.DataLoader(testset, batch_size=cifar5_batch_size, shuffle=False, num_workers=2)
            cifar5_testiter = iter(cifar5_testloader)

            ood_data_list = []

            while True:
                try:
                    cifar5_images, _ = cifar5_testiter.next()
                    ood_data_list.append(cifar5_images)
                except StopIteration:
                    break

            ood_data = torch.cat(ood_data_list, 0)
            return ood_data.numpy() * opt_config['ood_scale']

        elif ood_dataset_name == 'gaussian':
            return np.random.normal(size=(opt_config['n_anom'], 3, 32, 32)) * opt_config['ood_scale']
        elif ood_dataset_name == 'uniform':
            return np.random.uniform(size=(opt_config['n_anom'], 3, 32, 32)) * opt_config['ood_scale']


def load_ood_data(name_dataset, opt_config):
    n_anom = opt_config['n_anom']

    # Load OOD dataset
    if name_dataset == 'babymnist':
        # N_ANOM = 300
        pickle_file = './data/babynotMNIST.pickle'
        with open(pickle_file, 'rb') as f:
            ood_dataset = pickle.load(f)[:,None] * opt_config['ood_scale']  # (20000, 1, 8, 8)
        ood_dataset = Variable(torch.FloatTensor(ood_dataset[:n_anom]), volatile=True)
        return { 'notMNIST': ood_dataset }

    else:
        ood_data_dict = {}
        for ood_dataset_name in opt_config['ood_datasets']:
            ood_dataset = load_ood_dataset(name_dataset, ood_dataset_name, opt_config)
            ood_dataset = Variable(torch.FloatTensor(ood_dataset[:n_anom]), volatile=True)
            ood_data_dict[ood_dataset_name] = ood_dataset
        return ood_data_dict


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
        valset.train_data = valset.train_data[size_trainset:, :, :, :]
        valset.train_labels = valset.train_labels[size_trainset:]
        # Test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name_dataset == 'cifar5':
        # Just use images/labels that correspond to the first 5 classes: airplane, automobile, bird, cat, deer
        # Train set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        first_5_class_idxs = [i for i in range(len(trainset.train_labels)) if trainset.train_labels[i] in [0,1,2,3,4]]
        filtered_data = [trainset.train_data[i, :, :, :] for i in first_5_class_idxs]
        filtered_labels = [trainset.train_labels[i] for i in first_5_class_idxs]
        trainset.train_data = np.stack(filtered_data[:size_trainset])
        trainset.train_labels = np.stack(filtered_labels[:size_trainset])
        # Validation set
        valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valset.train_data = np.stack(filtered_data[size_trainset:])
        valset.train_labels = np.stack(filtered_labels[size_trainset:])
        # Test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        first_5_class_idxs = [i for i in range(len(testset.test_labels)) if testset.test_labels[i] in [0,1,2,3,4]]
        testset.test_data = np.stack([testset.test_data[i, :, :, :] for i in first_5_class_idxs])
        testset.test_labels = np.stack([testset.test_labels[i] for i in first_5_class_idxs])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, valloader, testloader, name_dataset



class MiniBatcher(object):
    def __init__(self, N, batch_size=32, loop=True, Y_semi=None, fraction_labelled_per_batch=None):
        self.N = N
        self.batch_size=batch_size
        self.loop = loop
        self.idxs = np.arange(N)
        np.random.shuffle(self.idxs)
        self.curr_idx = 0
        self.fraction_labelled_per_batch = fraction_labelled_per_batch
        if fraction_labelled_per_batch is not None:
            bool_labelled = Y_semi.numpy().sum(1) == 1
            self.labelled_idxs = np.nonzero(bool_labelled)[0]
            self.unlabelled_idxs = np.where(bool_labelled==0)[0]
            np.random.shuffle(self.labelled_idxs)
            np.random.shuffle(self.unlabelled_idxs)
            self.N_labelled = int(self.batch_size*self.fraction_labelled_per_batch)
            self.N_unlabelled = self.batch_size - self.N_labelled
            ### check if number of labels are enough, if not repeat labels
            if self.labelled_idxs.shape[0]<self.N_labelled:
                fac = np.ceil(self.N_labelled / self.labelled_idxs.shape[0])
                self.labelled_idxs = self.labelled_idxs.repeat(fac)
        self.start_unlabelled = 0
        self.start_unlabelled_train = 0

    def next(self, train_iter):
        if self.fraction_labelled_per_batch is None:
            if self.curr_idx+self.batch_size >= self.N:
                self.curr_idx=0
                if not self.loop:
                    return None
            ret = self.idxs[self.curr_idx:self.curr_idx+self.batch_size]
            self.curr_idx+=self.batch_size
            return ret
        else:
            # WARNING: never terminate (i.e. return None)
            np.random.shuffle(self.labelled_idxs)
            np.random.shuffle(self.unlabelled_idxs)
            return np.array(list(self.labelled_idxs[:self.N_labelled])+list(self.unlabelled_idxs[:self.N_unlabelled]))


def cuda(xs, arguments=None):
    if (torch.cuda.is_available() and (arguments is None or arguments['--cuda'])):
        if isinstance(xs, (list, tuple)):
            return [x.cuda() for x in xs]
        elif isinstance(xs, dict):
            newd = {}
            for k in xs:
                newd[k] = xs[k].cuda()
            return newd
        else:
            return xs.cuda()
    else:
        return xs


def check_nan(pytorch_var, check_big=False, message=""):
    """Check whether variable contains NaN and pdb stop.
    """

    if pytorch_var.ne(pytorch_var).data.any():
        if message:
            print(message)
        else:
            print("Stuck at NaN check")
        sys.stdout.flush();
        pdb.set_trace()

    if check_big and (pytorch_var.data.abs() > 10).any():
        if message:
            print(message)
        else:
            print("Stuck at big check")
        sys.stdout.flush();
        pdb.set_trace()
