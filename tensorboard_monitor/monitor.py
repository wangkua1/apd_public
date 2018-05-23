import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict


class Monitor(object):
    def __init__(self, name='log'):
        super(Monitor, self).__init__()
        # self.node_on_graph = tf.placeholder(tf.float32, name= name)
        # self.variable_in_computation = tf.summary.scalar(name, self.node_on_graph)

        self.dict_of_traces = defaultdict(list)
        self.name = name

    def record_matplot(self, data, iteration, writer, use_existing_key=True):
        if use_existing_key and writer not in self.dict_of_traces:
            raise Exception("recording something that's not initialized")
        self.dict_of_traces[writer].append((iteration, data))


    def save_plot_matplot(self, log_folder, iteration):
        plt.clf()
        plt.plot(zip(self.dict_of_traces['train_loss'])[0][0], zip(self.dict_of_traces['train_loss'])[1][0], 'b', label = 'training')
        plt.plot(zip(self.dict_of_traces['point_loss'])[0][0], zip(self.dict_of_traces['point_loss'])[1][0], 'g', label = 'point estimation')
        plt.plot(zip(self.dict_of_traces['bayesian_loss'])[0][0], zip(self.dict_of_traces['bayesian_loss'])[1][0], 'r', label = 'bayesian')

        plt.xlabel('iteration')
        plt.ylabel(self.name)
        if self.name == 'cross_entropy':
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='lower right')
        plt.ylim([0,3])
        plt.savefig(os.path.join(log_folder, self.name + '_%g.png' % iteration))

    def load(self, log_folder):
        self.dict_of_traces = dict(np.load(os.path.join(log_folder, self.name+'.npz')))
        for k, v in self.dict_of_traces.items(): self.dict_of_traces[k] = v.T
    def save_result_numpy(self, log_folder):
        outfile = os.path.join(log_folder, self.name)
        np.savez(outfile, **self.dict_of_traces)
