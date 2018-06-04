"""adv_eval_new.py
Usage:
    adv_eval_new.py -g <exp_name> <attack> [--cuda]
    adv_eval_new.py -g <exp_name> <gan_dir> <attack> --gan [--cuda]
    adv_eval_new.py --sgld <exp_name> <attack> <num_samples> [--atk_source=<dir>] [--cuda]
    adv_eval_new.py --dropout <exp_name> <attack> <num_passes> [--atk_source=<dir>] [--cuda]
    adv_eval_new.py --gan <exp_name> <gan_dir> <attack> <num_samples> [--atk_source=<dir>] [--cuda]
    
Options:
    --cuda              Use cuda  [default: True]
    --atk_source=<dir>  Location of attacks [default: local]

Arguments:

Example:
    # Generate Adversarial Examples:
    python adv_eval_new.py -g fc1-mnist-100-X-sgld-mnist-1-X-mnist-20000@2018-01-23 pgd 5000 --cuda
    # Evaluate Adversarial Examples using SGLD samples
    python adv_eval_new.py --sgld fc1-mnist-100-X-sgld-mnist-1-X-mnist-20000@2018-01-23 pgd 5000 --cuda
    # Evaluate Adversarial Examples with MC Dropout
    python adv_eval_new.py --dropout fc1-mnist-100-drop-50-X-sgd-mnist-X-mnist-20000@2018-01-23 pgd 5000 --cuda
    # Evaluate Adversarial Examples with GAN
    python adv_eval_new.py --gan fc1-mnist-100-drop-50-X-sgd-mnist-X-mnist-20000@2018-01-23 gan_exps/_mnist-wgan-gp-1000100 pgd 5000 --cuda


Comments:
    Runs adversarial evaluation

--sgld
--mcmc
--gan
"""

import os, errno

import utils
from utils import load_posterior_samples
from opt.loss import *
from model.fc import *
from pgd import FBOXPGDAttack

import yaml
from docopt import docopt

import sklearn
import numpy as np
import torch
from torch.autograd import Variable

from gan_pytorch import Generator

import foolbox

TOTAL=6000

def get_model(exp_name, arguments):
    model_conf = 'model/config/' + exp_name[:exp_name.find('X')-1] + '.yaml'
    model_config = yaml.load(open(model_conf, 'rb'))
    model = eval(model_config['name'])(**model_config['kwargs'])
    model.load_state_dict(torch.load('saves/' + exp_name + '/best_point_model.th'))
    return utils.cuda(model, arguments)

def get_generator(gan_dir, arguments):
    gan_config = yaml.load(open(os.path.join(gan_dir, 'config.yaml')))
    zdim = int(gan_config['zdim'])
    hdim = int(gan_config['gan_dim'])
    odim = int(gan_config['output_dim'])
    generator = utils.cuda(Generator(zdim, hdim, odim), arguments)
    generator.load_state_dict(torch.load(os.path.join(gan_dir, 'generator.pt')))
    return generator

def generate_samples(generator, num_samples, batch_size=20):
    iters = int(num_samples / batch_size) + 1
    samples = []
    for _ in range(iters):
        p = generator.generate(batch_size).data.cpu().numpy()
        samples.append(p)
    all_samples = np.concatenate(samples, 0)
    return all_samples[:num_samples]

def get_opt_config(exp_name):
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
    second_parts = exp_name[exp_name.find('X') + 2:]
    opt_config_name = second_parts[:second_parts.find('X') - 1]
    opt_config = yaml.load(open('opt/config/' + opt_config_name + '.yaml', 'rb'))

    # Fill in default configuration for keys that are not overwritten by the config file
    for key in default_config:
        if key not in opt_config:
            opt_config[key] = default_config[key]
    
    return opt_config

def trim_dataloader(dataloader, arguments, total=TOTAL):
    inputs = []
    sub_total = 0
    for test_inputs, test_labels in dataloader:
        sub_total += test_inputs.size(0)
        test_inputs, test_labels = utils.cuda((test_inputs, test_labels), arguments)
        test_inputs = Variable(test_inputs, volatile=True)
        inputs.append(test_inputs)
        if sub_total > total:
            break
    return torch.cat(inputs, 0)[:TOTAL]

def get_dataset_id(exp_name):
    second_parts = exp_name[exp_name.find('X') + 2:]
    third_parts = second_parts[second_parts.find('X') + 2:]
    return third_parts[:third_parts.find('@')]

def gen_adv_examples(model, attack, arguments, total=TOTAL):
    model.eval()
    fb_model = foolbox.models.PyTorchModel(model, (-1,1), 10, cuda=arguments['--cuda'])
    attack_instance = attack(fb_model)

    # Lousy programmer retrieve dataset
    exp_name = arguments['<exp_name>']
    dataset_id = get_dataset_id(exp_name)
    _, valloader, _, _ = utils.get_dataloader(dataset_id, 1)

    ad_labels = []
    true_labels = []
    adv_examples = []

    for data, label in valloader:
        if len(adv_examples) == total:
            break
        # import pdb; pdb.set_trace()
        label = label.type(torch.LongTensor)
        adversarial = attack_instance(data.numpy()[0], label=label.numpy()[0])
        if adversarial is not None:
            adv_examples.append(adversarial)
            adv_ex = Variable(torch.Tensor(adversarial))
            if arguments['--cuda']:
                adv_ex = adv_ex.cuda()
            ad_label = model(adv_ex)
            ad_labels.append(ad_label.data.cpu().numpy())
            true_labels.append(label.numpy())
    print("Adv Fail Rate: {}".format(np.mean(np.array(ad_labels) == np.array(true_labels))))
    return np.array(adv_examples), np.array(ad_labels), np.array(true_labels)

def run_adv_detection(adv_examples, samples=None, f_pred=utils.posterior_uncertainty, f_acq='f_identity', **kwargs):
    model = kwargs['model']
    if samples is not None:
        model.posterior_samples = utils.prepare_torch_dicts(samples, model)
        model.posterior_weights = [1 for _ in range(len(model.posterior_samples))]

    _, _, testloader, _ = utils.get_dataloader('mnist-40000', 200)
    test_inputs = trim_dataloader(testloader, None, adv_examples.shape[0])

    normality_base_rate, auroc, n_aupr, ab_aupr = utils.show_ood_detection_results_softmax(test_inputs, adv_examples, f_pred,
                                                                                                                   kwargs, f_acq)
    print(
    "(Anomaly Detection Results: \nBase Rate: {:.2f}, AUROC: {:.2f}, AUPR+: {:.2f}, AUPR-: {:.2f}".format(
        normality_base_rate, auroc, n_aupr, ab_aupr))
    return normality_base_rate, auroc, n_aupr, ab_aupr

def uncertainty_hist(model, adv_ex, samples):
    model.posterior_samples = utils.prepare_torch_dicts(samples, model)
    model.posterior_weights = [1 for _ in range(len(model.posterior_samples))]

    adv_uncert = utils.posterior_uncertainty(model, adv_ex).data.cpu().numpy()
    _, _, testloader, _ = utils.get_dataloader('mnist-40000', 200)

    test_uncert = []
    for data, label in testloader:
        data = Variable(data)
        if arguments['--cuda']:
            data = data.cuda()
        test_uncert.append(utils.posterior_uncertainty(model, data).data.cpu().numpy())
    test_uncert = np.concatenate(test_uncert, 0)

    import matplotlib.pyplot as plt
    test_density, test_edges = np.histogram(test_uncert, 50, range=(0.0, 0.05))
    test_bin = [(test_edges[i] + test_edges[i+1]) / 2.0 for i in range(len(test_edges) - 1)]
    
    adv_density, adv_edges = np.histogram(adv_uncert, 50, range=(0.0, 0.05))
    adv_bin = [(adv_edges[i] + adv_edges[i+1]) / 2.0 for i in range(len(adv_edges) - 1)]

    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    l1, = plt.plot(test_bin, test_density, label='Test Set', lw=4)
    l2, = plt.plot(adv_bin, adv_density, label='Adversarial', lw=4)
    
    plt.legend(handles=[l1, l2])
    plt.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the left edge are off
        right='off',         # ticks along the right edge are off
        labelleft='off') # labels along the left edge are off
    plt.title('Uncertainty Histogram of Test Data vs. Adv. Examples', size=16)
    plt.xlabel('Uncertainty', size=14)
    plt.ylabel('Frequency', size=14)
    plt.ylim(0.0, 100)
    plt.tight_layout()
    plt.show()

def load_adv_examples(exp_name, attack_name, arguments):
    path = os.path.join('saves', exp_name, 'adv', attack_name, 'examples.npy')
    if os.path.exists(path):
        return Variable(utils.cuda(torch.Tensor(np.load(path)), arguments), volatile=True)
    else:
        raise Exception('%s adv examples must be generated first (use -g flag)' % attack_name)

def main(arguments):
    exp_name = arguments['<exp_name>']
    model = get_model(exp_name, arguments)

    # Generate examples
    if arguments['-g']:
        attack_name = arguments['<attack>']
        adv_path_prefix = os.path.join('saves', exp_name, 'adv', attack_name)
        if arguments['--gan']:
            gan_dir = arguments['<gan_dir>']
            adv_path_prefix = os.path.join('saves', gan_dir, 'adv', attack_name)
            generator = get_generator(gan_dir, arguments)
            post_samples = generate_samples(generator, 1)
            model.load_state_dict(utils.prepare_torch_dicts(post_samples, model)[0])
        elif 'drop' not in exp_name:
            post_samples = np.array(load_posterior_samples(exp_name, 1))
            model.load_state_dict(utils.prepare_torch_dicts(post_samples, model)[0])

        if attack_name.lower() == 'fgsm':
            attack = foolbox.attacks.GradientSignAttack
        elif attack_name.lower() == 'lbfgs':
            attack = foolbox.attacks.LBFGSAttack
        elif attack_name.lower() == 'pgd':
            attack = FBOXPGDAttack
        else:
            raise NotImplementedError('Unsupported attack type %s' % attack_name)
        adv_examples, adv_labels, true_labels = gen_adv_examples(model, attack, arguments)
        print("Found {} adversarial examples".format(adv_examples.shape[0]))
        try:
            os.makedirs(adv_path_prefix)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        np.save(adv_path_prefix + '/examples', adv_examples)
        np.save(adv_path_prefix + '/adv_labels', adv_labels)
        np.save(adv_path_prefix + '/true_labels', true_labels)
    else:
        atk_source = arguments['--atk_source']
        f_pred = utils.posterior_expectation
        acq_funcs = ['f_entropy', 'f_bald', 'f_uncert_x']
        attack_name = arguments['<attack>']
        print('Loading adv examples from {}'.format(exp_name if atk_source == 'local' else atk_source))
        if atk_source == 'local':
            adv_examples = load_adv_examples(exp_name, attack_name, arguments)
        else:
            adv_examples = load_adv_examples(atk_source, attack_name, arguments)
        # Evaluate attack using SGLD sampling
        if arguments['--sgld']:
            num_samples = int(arguments['<num_samples>'])

            post_samples = np.array(load_posterior_samples(exp_name, num_samples))
            for f_acq in acq_funcs:
                keep_samples = False
                if f_acq == 'f_bald' or f_acq == 'f_uncert_x':
                    keep_samples = True
                print('Acquisition function: {}'.format(f_acq))
                run_adv_detection(adv_examples, post_samples, f_pred, f_acq,
                                model=model, keep_samples=keep_samples, use_mini_batch=200)
        # Evaluate attack using MCDropout
        elif arguments['--dropout']:
            if 'drop' not in exp_name:
                raise Exception('Must use dropout model')
            num_passes = int(arguments['<num_passes>'])

            f_pred = utils.mc_dropout_expectation
            for f_acq in acq_funcs:
                keep_samples = False
                if f_acq == 'f_bald'  or f_acq == 'f_uncert_x':
                    keep_samples = True
                print('Acquisition function: {}'.format(f_acq))
                run_adv_detection(adv_examples, None, f_pred, f_acq,
                                model=model, passes=num_passes, keep_samples=keep_samples)
        elif arguments['--gan']:
            gan_dir = arguments['<gan_dir>']
            # if arguments['--atk_source'] == 'local':
            #     load_adv_examples(gan_dir, attack_name, arguments)

            num_samples = int(arguments['<num_samples>'])
            generator = get_generator(gan_dir, arguments)

            gan_samples = generate_samples(generator, num_samples)
            for f_acq in acq_funcs:
                keep_samples = False
                if f_acq == 'f_bald'  or f_acq == 'f_uncert_x':
                    keep_samples = True
                print('Acquisition function: {}'.format(f_acq))
                run_adv_detection(adv_examples, gan_samples, f_pred, f_acq,
                                model=model, keep_samples=keep_samples, use_mini_batch=200)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt...")
    print(arguments)
    print("............\n")

    main(arguments)
