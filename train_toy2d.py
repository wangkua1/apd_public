"""train_toy2d.py
Usage:
    train_toy2d.py <f_model_config> <f_opt_config>  [--prefix <p>] [--ce] [--db]
    train_toy2d.py -r <exp_name> <idx> [--test]

Arguments:
 
Options:
"""
from __future__ import division
import matplotlib as mtl
mtl.use('Agg')
import copy
from docopt import docopt
import yaml
import torch
import tensorflow as tf
from torch import optim
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
import numpy as np

from utils import MiniBatcher#, MiniBatcherPerClass
import datetime
from opt.loss import *
from opt.nsgd import NoisedSGD
from model.fc import fc
# from model.cnn import *
import cPickle as pkl
import itertools
import matplotlib.pyplot as plt
#### magic
tft = lambda x:torch.FloatTensor(x)
tfv = lambda x:Variable(tft(x))


arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")




if arguments['-r']:
    exp_name = arguments['<exp_name>']
    f_model_config = 'model/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[0]+'.yaml'
    f_opt_config = 'opt/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[1]+'.yaml'
    old_exp_name = exp_name
    exp_name += '_resumed'
else:
    f_model_config = arguments['<f_model_config>']
    f_opt_config = arguments['<f_opt_config>']
    model_name = os.path.basename(f_model_config).split('.')[0]
    opt_name = os.path.basename(f_opt_config).split('.')[0]
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    data_name = 'mnist'
    if arguments['--prefix']:
        exp_name = '%s:%s-X-%s-X-%s@%s' % ('toy', model_name, opt_name, data_name, timestamp)
    else:
        exp_name = '%s-X-%s-X-%s@%s' % (model_name, opt_name, data_name, timestamp)
    if arguments['--ce']:
        exp_name = 'CE.' + exp_name
    

model_config = yaml.load(open(f_model_config, 'rb'))
opt_config = yaml.load(open(f_opt_config, 'rb'))

print ('\n\n\n\n>>>>>>>>> [Experiment Name]')
print (exp_name)
print ('<<<<<<<<<\n\n\n\n')

## Experiment stuff
if not os.path.exists('./saves/%s'%exp_name):
    os.makedirs('./saves/%s'%exp_name)

## Data
size = 10
m1 = [-2,-2]
m2 = [2,2]
cov = np.eye(2) * .3
x1 = np.random.multivariate_normal(m1, cov,size=size)
x2 = np.random.multivariate_normal(m2, cov,size=size)
X = np.vstack([x1,x2])
Y = np.zeros((size*2, 2))
Y[:size,0]=1
Y[size:,1]=1

X = tfv(X)
Y = tft(Y)

linspace = np.arange(-5,5,0.1)
test_points = np.array(list(itertools.product(linspace, linspace)))


def _plot(test_points, py1_xs ):
    im = np.zeros((99,99))
    for (py1_x, tp) in zip(py1_xs, test_points):
        row, col = int(tp[0]*10+50), int(tp[1]*10+50)
        im[row, col] = py1_x
    return im
# Dataset (X size(N,D) , Y size(N,K))
## Model
model = eval(model_config['name'])(**model_config['kwargs'])
model.type(torch.FloatTensor)
## Optimizer
opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])


## tensorboard
#ph
ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
ph_loss = tf.placeholder(tf.float32,  name='loss')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
tf_acc = tf.summary.scalar('accuracy', ph_accuracy)
tf_loss = tf.summary.scalar('loss', ph_loss)

log_folder = os.path.join('./logs', exp_name)
# remove existing log folder for the same model.
if os.path.exists(log_folder):
    import shutil
    shutil.rmtree(log_folder, ignore_errors=True)

sess = tf.InteractiveSession()   

train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)

batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

## Loss
if arguments['--ce']:
    Loss = CE()
else:
    raise NotImplementedError()


best_val_acc = 0
val_errors = []
tf.global_variables_initializer().run()
posterior_samples =[]
posterior_weights = []
is_collecting = True
# alpha_thresh = 0.01
sample_size = 1000
sample_interval = 20
plt.clf()
if not arguments['--db']:
    ## Algorithm
    sd = opt.state_dict()
    step_size = sd['param_groups'][0]['lr']     
    for idx in tqdm(xrange(opt_config['max_train_iters'])):
    # for idx in (xrange(opt_config['max_train_iters'])):
        if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == idx:
            _, tmp_fac = opt_config['lrsche'].pop(0)
            sd = opt.state_dict()
            assert len(sd['param_groups']) ==1
            sd['param_groups'][0]['lr'] *= tmp_fac
            opt.load_state_dict(sd)
        if idx > 0 and  'lrpoly' in opt_config :
            a, b, g = opt_config['lrpoly']
            sd = opt.state_dict()
            step_size = a*((b+idx)**(-g))
            sd['param_groups'][0]['lr'] = step_size
            opt.load_state_dict(sd)

        idxs = batcher.next(idx)
        X_batch = X[torch.LongTensor(idxs)].type(torch.FloatTensor)
        Y_batch = Y[torch.LongTensor(idxs)]#.type(torch.FloatTensor)
        ## network
        tv_F = model.forward(X_batch)
        F = tv_F.data.clone().type(torch.FloatTensor)
        ### loss layer
        loss, G, train_pred = Loss.train(F, Y_batch)

        model.zero_grad()
        tv_F.backward(gradient=G.type(torch.FloatTensor))
        opt.step()


        # TensorBoard
        #accuracy
        train_gt = Y[torch.LongTensor(idxs)].numpy().argmax(1)
        train_accuracy = (train_pred[batcher.start_unlabelled:] == train_gt[batcher.start_unlabelled:]).mean()

        # summarize
        acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_accuracy})
        loss = sess.run(tf_loss, feed_dict={ph_loss:loss})
        tmp = Y_batch.numpy()
        train_writer.add_summary(acc+loss, idx)


        if is_collecting  and idx%sample_interval==0: 
            posterior_samples.append(copy.deepcopy(model.model.state_dict()))
            posterior_weights.append(step_size)
            if len(posterior_samples) > sample_size:
                del posterior_samples[0]
                del posterior_weights[0]

        #validate
        if idx>0 and idx%100==0:
            curr_state = model.model.state_dict()
            def _validate_batch(model, X_val_batch):
                model.eval()
                _, proba = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor), ret_proba=True)
                model.train()
                return proba[:,0]
            
            non_bayes_probas = _validate_batch(model, test_points)
            
            def _validate_batch_bayes(posterior_samples,posterior_weights, X_val_batch):
                model.eval()
                acc_proba = None
                for sample_idx in xrange(len(posterior_samples)):
                    p_sample = posterior_samples[sample_idx]
                    model.model.load_state_dict(p_sample)
                    _,proba = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor), ret_proba=True)
                    if acc_proba is None:
                        acc_proba = posterior_weights[sample_idx] * proba
                    else:
                        acc_proba += posterior_weights[sample_idx] * proba
                model.train()
                acc_proba /= sum(posterior_weights)
                return acc_proba[:,0]
            if is_collecting:
                    bayes_probas = _validate_batch_bayes(posterior_samples[-sample_size:],posterior_weights[-sample_size:], test_points)
            npX = X.data.numpy()
            plt.scatter(npX[:size,0]*10+50,npX[:size,1]*10+50,c='k')
            plt.scatter(npX[size:,0]*10+50,npX[size:,1]*10+50,c='w')
            
            plt.imshow(_plot(test_points, bayes_probas), cmap=plt.cm.rainbow,interpolation='bicubic')
            plt.savefig(os.path.join(log_folder, 'bayes_probas_%g.png'%idx))
            plt.imshow(_plot(test_points, non_bayes_probas), cmap=plt.cm.rainbow,interpolation='bicubic')
            plt.savefig(os.path.join(log_folder, 'non_bayes_probas_%g.png'%idx))

            model.model.load_state_dict(curr_state)
        ## checkpoint
        if idx>0 and idx%100==0:
            name = './saves/%s/model_%i.t7'%(exp_name,idx)
            print ("[Saving to]")
            print (name)
            model.save(name)
            torch.save(opt.state_dict(), './saves/%s/opt_%i.t7'%(exp_name,idx))
        if idx>0 and idx%(sample_size*sample_interval)==0:
            def _flatten_npyfy(posterior_samples):
                return np.array([np.concatenate([p.numpy().ravel() for p in sample.values()]) for sample in posterior_samples])
            np.save('./saves/%s/params_%i'%(exp_name, idx//(sample_size*sample_interval)),_flatten_npyfy(posterior_samples))
pkl.dump(val_errors, open(os.path.join(log_folder, 'val.log'), 'wb'))



