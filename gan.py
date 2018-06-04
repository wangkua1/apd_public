"""gan.py

Usage:
    gan.py <src_dir> <f_opt_config> [--cuda] [--test]

Example:
    # FORMAT: Pass in the save directory and the GAN config file
    python gan.py fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@2018-05-31 opt/gan-config/gan1.yaml
"""

import os
import sys
import pdb
import time
import yaml
import itertools
sys.path.append(os.getcwd())
from docopt import docopt
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tflib as lib
import tflib.ops.linear
# import tflib.ops.batchnorm
import tflib.plot
# from tqdm import tqdm

import torch

import matplotlib.pyplot as plt

# Local imports
import utils
import gan_utils
from opt.loss import *


"""
tensorflow stuff
"""
def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
        # initialization=('uniform', 0.1)
    )
    output = tf.nn.relu(output)
    return output


def Generator(n_samples):
    output = tf.random_normal([n_samples, ZDIM])
    output = ReLULayer('Generator.1', ZDIM, DIM, output)
    output = ReLULayer('Generator.2', DIM, DIM, output)
    output = ReLULayer('Generator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Generator.4', DIM, OUTPUT_DIM, output)
    return output


def Discriminator(inputs):
    output = inputs
    output = ReLULayer('Discriminator.1', OUTPUT_DIM, DIM, output)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

"""
END of tensorflow stuff
"""


if __name__ == '__main__':

    try:
        arguments = docopt(__doc__)
    except: ## TODO: tmp using this to enable calling from another file
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

    if exp_dir_prefix:
        exp_dir = '{}-{}-{}-{}-{}-{}'.format(exp_dir_prefix, TASK, MODE, opt_config['n_train'], DIM, opt_config['ood_scale'])
    else:
        exp_dir = '{}-{}-{}-{}-{}'.format(TASK, MODE, opt_config['n_train'], DIM, opt_config['ood_scale'])

    ZDIM = opt_config['zdim']  # The noise dimension of the generator
    LAMBDA = opt_config['lambda'] # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = opt_config['critic_iters'] # How many critic iterations per generator iteration
    BATCH_SIZE = opt_config['batch_size'] # Batch size
    # TESTING = False


    if arguments['--test']:
        ITERS = 0  # Don't do any training iterations, just jump to the test code
    else:
        ITERS = opt_config['iters'] # How many generator iterations to train for


    # Dataset iterators
    train_gen = gan_utils.data_generator(src_dir, BATCH_SIZE, N_TRAIN=opt_config['n_train'], sampling_type=opt_config['sampling_type'])
    OUTPUT_DIM = list(train_gen())[0].shape[1]

    if not os.path.exists(os.path.join('gan_exps',exp_dir)):
        os.makedirs(os.path.join('gan_exps',exp_dir))


    real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE)

    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

        clip_ops = []
        for var in disc_params:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
        disc_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name('Generator'))
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name('Discriminator.'))


    lib.print_model_settings(locals().copy())


    log_dir = os.path.abspath(os.path.join('gan_exps', exp_dir))

    model = gan_utils.model_load_configuration(arguments)
    if arguments['--cuda']:
        model.cuda()  # Move the model onto the GPU

    if TASK == 'toy2d':
        validater = gan_utils.EvalToy2d(model)
        task_fs = [validater.toy2d_validate]
    elif TASK == 'babymnist':
        validater = gan_utils.EvalBabyMNIST(model, opt_config, arguments['--cuda'], log_dir)
        task_fs = [validater.babymnist_validate, validater.babymnist_ood]
    elif TASK == 'mnist':
        validater = gan_utils.EvalMNIST(model, opt_config, arguments['--cuda'], log_dir)
        task_fs = [validater.mnist_validate, validater.mnist_ood]
    elif TASK == 'cifar5':
        validater = gan_utils.EvalCIFAR(model, opt_config, arguments['--cuda'], log_dir)
        task_fs = [validater.cifar_validate, validater.cifar_ood]
    else:
        raise NotImplementedError()

    saver = tf.train.Saver()

    os.chdir(os.path.join('gan_exps', exp_dir))
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        gen = gan_utils.inf_train_gen(train_gen)

        ################
        ### Training ###
        ################

        # At any point you can hit Ctrl+C to break out of training early, and proceed to run on test data.
        try:

            for iteration in range(ITERS):
                start_time = time.time()
                # Train generator
                if iteration > 0:
                    _ = session.run(gen_train_op)
                # Train critic
                if MODE == 'dcgan':
                    disc_iters = 1
                else:
                    disc_iters = CRITIC_ITERS

                for i in range(disc_iters):
                    # _data = gen.next()
                    _data = next(gen)
                    _disc_cost, _, grads = session.run([disc_cost, disc_train_op, gradients], feed_dict={real_data: _data})

                    if MODE == 'wgan':
                        _ = session.run(clip_disc_weights)

                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('time', time.time() - start_time)


                # # Calculate dev loss and generate samples every 100 iters
                # if iteration % 100 == 99:
                #     dev_disc_costs = []
                #     for images,_ in dev_gen():
                #         _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images})
                #         dev_disc_costs.append(_dev_disc_cost)
                #     lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                #     generate_image(iteration, _data)

                # Save logs every 100 iters

                # if (iteration < 5) or (iteration % 100 == 99):
                # if (iteration < 5) or (iteration % opt_config['validation_interval'] == 0):
                if (iteration > 5) and (iteration % opt_config['validation_interval'] == 0):
                    lib.plot.flush()

                    # saver.save(session, 'gan_checkpoint', global_step=iteration)

                    sample_params = session.run(fake_data)
                    sample_dics = utils.prepare_torch_dicts(sample_params, model)
                    sample_dics_real = utils.prepare_torch_dicts(_data, model)

                    print("Fake")
                    print("----")
                    for task_f in task_fs:
                        task_f(sample_dics, iteration, log='fake')

                    print("=" * 50)

                    print("Real")
                    print("----")
                    for task_f in task_fs:
                        task_f(sample_dics_real, iteration, log='real')


                    print()

                lib.plot.tick()
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early!')



        ########################
        ### GAN TEST RESULTS ###
        ########################

        os.chdir('../..')

        Loss = CE()
        testloader = validater.testloader
        ood_data = validater.ood_data

        num_test_runs = opt_config['num_test_runs']
        num_test_samples = opt_config['num_test_samples']
        # num_samples = BATCH_SIZE

        ###########################
        ### Test Classification ###
        ###########################
        posterior_flag = 1  # Hardcoded for now

        gan_accuracy_list = []
        for i in range(num_test_runs):
            model.eval()

            print("GAN Samples | Test run {}".format(i))

            # sample_params = session.run(fake_data)

            sample_params = []
            for _ in range(num_test_samples // BATCH_SIZE + 1):
                sample_params.append(session.run(fake_data))
            sample_params = np.concatenate(sample_params, 0)
            sample_params = sample_params[:num_test_samples]

            sample_dics = utils.prepare_torch_dicts(sample_params, model)
            posterior_weights = [1 for _ in range(len(sample_dics))]

            model.posterior_samples = sample_dics  # Should change this structure
            model.posterior_weights = posterior_weights
            point_accuracy, point_loss, posterior_accuracy, posterior_loss = gan_utils.evaluate(model, testloader, sample_dics, posterior_weights, posterior_flag, Loss, opt_config, arguments)
            print("Posterior acc: {}".format(posterior_accuracy))
            gan_accuracy_list.append(posterior_accuracy)

        print("GAN Classification Results")
        print("--------------------------")
        print("Test Accuracy: Mean {}, Std {}\n".format(np.mean(gan_accuracy_list), np.std(gan_accuracy_list)))

        real_accuracy_list = []
        for i in range(num_test_runs):
            print("True Samples | Test run {}".format(i))

            sample_dics_real = utils.load_posterior_state_dicts(src_dir=src_dir, example_model=model, num_samples=num_test_samples)

            posterior_weights = [1 for _ in range(len(sample_dics_real))]
            model.posterior_samples = sample_dics_real  # Should change this structure
            model.posterior_weights = posterior_weights
            point_accuracy, point_loss, posterior_accuracy, posterior_loss = gan_utils.evaluate(model, testloader, sample_dics_real, posterior_weights, posterior_flag, Loss, opt_config, arguments)
            print("Posterior acc: {}".format(posterior_accuracy))
            real_accuracy_list.append(posterior_accuracy)

        print("Real Sample Classification Results")
        print("----------------------------------")
        print("Test Accuracy: Mean {}, Std {}\n".format(np.mean(real_accuracy_list), np.std(real_accuracy_list)))


        ##############################
        ### Test Anomaly detection ###
        ##############################

        for scale in opt_config['test_ood_scales']:
            model.eval()  # Just to make sure model is in eval mode

            scale = int(scale)

            # Temporarily reset ood_scale in opt_config for simplicity with the rest of the code
            opt_config['ood_scale'] = scale
            ood_data = utils.load_ood_data(opt_config['task'], opt_config)

            print("OOD SCALE {}".format(scale))
            print("--------------------------")


            test_inputs_anomaly_detection = utils.get_anomaly_detection_test_inputs(testloader, opt_config, arguments)
            if arguments['--cuda']:
                test_inputs_anomaly_detection = test_inputs_anomaly_detection.cuda()
                for key in ood_data:
                    ood_data[key] = ood_data[key].cuda()

            print("GAN OOD Results")
            print("---------------")

            for ood_dataset_name in opt_config['ood_datasets']:
                print("OOD Dataset: {}".format(ood_dataset_name))

                cur_ood_data = ood_data[ood_dataset_name]

                for func_name in opt_config['ood_acq_funcs']:
                    normality_base_rate_list = []
                    auroc_list = []
                    n_aupr_list = []
                    ab_aupr_list = []

                    for i in range(num_test_runs):
                        # print("Test run {}".format(i))

                        sample_params = []
                        for _ in range(num_test_samples // BATCH_SIZE + 1):
                            sample_params.append(session.run(fake_data))
                        sample_params = np.concatenate(sample_params, 0)
                        sample_params = sample_params[:num_test_samples]

                        sample_dics = utils.prepare_torch_dicts(sample_params, model)
                        posterior_weights = [1 for _ in range(len(sample_dics))]

                        model.posterior_samples = sample_dics  # Should change this structure
                        model.posterior_weights = posterior_weights

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


            print
            print("Real OOD Results")
            print("----------------")

            for ood_dataset_name in opt_config['ood_datasets']:
                print("OOD Dataset: {}".format(ood_dataset_name))

                cur_ood_data = ood_data[ood_dataset_name]

                for func_name in opt_config['ood_acq_funcs']:

                    normality_base_rate_list = []
                    auroc_list = []
                    n_aupr_list = []
                    ab_aupr_list = []

                    for i in range(num_test_runs):
                        # print("Test run {}".format(i))
                        posterior_samples = utils.load_posterior_state_dicts(src_dir=src_dir, example_model=model, num_samples=num_test_samples)
                        posterior_weights = [1 for _ in range(len(posterior_samples))]
                        model.posterior_samples = posterior_samples  # Should change this structure
                        model.posterior_weights = posterior_weights

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
