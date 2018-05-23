import pdb

import torch
from torch.autograd import Variable
import numpy as np


class CE(object):
    """docstring for CE
    standard classification loss
    """
    def __init__(self):
        super(CE, self).__init__()
        self.lsoftmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.Softmax()
        self.nll = torch.nn.NLLLoss()
        self.CEL = torch.nn.CrossEntropyLoss()

    def CrossEntropyLoss(self, outputs, labels):
        return self.CEL(outputs.type(torch.FloatTensor), labels)

    def softmax_output(self, inputs):  # Why not just use F.softmax(inputs) ?
        prob_inputs = self.softmax(inputs)
        return prob_inputs

    def log_softmax_output(self, inputs):  # Use F.log_softmax(inputs)
        prob_inputs = self.lsoftmax(inputs)
        return prob_inputs

    def inference_prediction(self, inputs):
        prob_inputs = self.softmax(inputs)
        prediction = prob_inputs.data.cpu().numpy().argmax(1)
        return prediction

    def cross_entropy_loss(self, inputs, label):
        # pdb.set_trace()
        prob_inputs = self.softmax(inputs)
        cross_e = self.CrossEntropyLoss(prob_inputs, label)
        return cross_e

    def nll_loss(self, inputs, label):
        prob_inputs = self.lsoftmax(inputs.type(torch.FloatTensor))
        nll = self.nll(prob_inputs, label)
        return nll

    def train(self, F, Y_batch):
        tv_F = Variable(F, requires_grad=True)
        tv_Y = Variable(torch.LongTensor(Y_batch.numpy().argmax(1)))
        py_x = self.lsoftmax(tv_F)
        loss = self.nll(py_x, tv_Y)
        ##
        loss.backward()
        G = tv_F.grad.data
        train_pred = py_x.data.numpy().argmax(1)
        return loss.data[0], G, train_pred

    def infer(self, model, X_val, ret_proba=False):
        py_x = self.softmax(model.forward(X_val))
        proba = py_x.data.cpu().numpy()
        val_pred = proba.argmax(1)
        if ret_proba:
            return val_pred, proba
        else:
            return val_pred
