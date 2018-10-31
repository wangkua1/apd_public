# Adversarial Posterior Distillation (APD)

This repository contains the code used for the paper [Adversarial Distillation of Bayesian Neural Network Posteriors (ICML 2018)](https://arxiv.org/pdf/1806.10317.pdf).

## Requirements

* Python 3.6.3
* PyTorch 0.3.1.post2
* Tensorflow 1.4.0


## Environment Setup

Here is an example that shows how to set up a conda environment with the appropriate versions of the frameworks:

```
conda create -n apd-env python=3.6
source activate apd-env
conda install pytorch=0.3.1 cuda80 -c soumith
conda install torchvision -c pytorch
pip install -r requirements.txt
```


## Experiments

### Toy 2D Classification



### Predictive Performance and Uncertainty

#### MNIST fcNN1 (784-100-10)

**SGD**
```
python train_new.py model/config/fc1-mnist-100.yaml opt/config/sgd-mnist.yaml mnist-50000 --cuda
```

**MC-Dropout (p=0.5)**
```
python train_new.py model/config/fc1-mnist-100-drop-50.yaml opt/config/sgd-mnist.yaml mnist-50000 --cuda --mc_dropout_passes 200
```

**SGLD**
```
python train_new.py model/config/fc1-mnist-100.yaml opt/config/sgld-mnist-1-1.yaml mnist-50000 --cuda
```

**APD**
```
python gan.py fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE opt/gan-config/gan1.yaml
```


#### MNIST fcNN2 (784-400-400-10)

**SGD**
```
python train_new.py model/config/fc-mnist-400.yaml opt/config/sgd-mnist.yaml mnist-50000 --cuda
```

**MC-Dropout (p=0.5)**
```
python train_new.py model/config/fc-mnist-400-drop-50.yaml opt/config/sgd-mnist.yaml mnist-50000 --cuda --mc_dropout_passes 200
```

**SGLD**
```
python train_new.py model/config/fc-mnist-400.yaml opt/config/sgld-mnist-1-1.yaml mnist-50000 --cuda
```

**APD**
```
python gan.py fc-mnist-400-X-sgld-mnist-1-1-X-mnist-50000@DATE opt/gan-config/gan1.yaml
```


### Active Learning

See commands in paper-act2.sh for running active learning experiments.

There is a ipynb in notebooks for visualizing the results.


### Adversarial Example Detection

These experiments require the installation of [foolbox](https://github.com/bethgelab/foolbox/tree/master/foolbox).

First, train the SGLD and MC-Dropout networks as above:

**MC-Dropout (p=0.5)**
```
python train_new.py model/config/fc1-mnist-100-drop-50.yaml opt/config/sgd-mnist.yaml mnist-50000 --cuda --mc_dropout_passes 200
```


**SGLD**
```
python train_new.py model/config/fc1-mnist-100.yaml opt/config/sgld-mnist-1.yaml mnist-50000 --cuda
```


**APD**

Training APD requires the `gan_pytorch.py` script:

```
python gan_pytorch.py fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE opt/gan-config/gan1.yaml --cuda
```


#### Generating Adversarial Examples

Next, generate the adversarial examples for each model. Replace `fgsm` with `pgd` for the PGD attack.

**SGLD**

```
python adv_eval_new.py -g fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE fgsm --cuda
```

**MC-Dropout (p=0.5)**

```
python adv_eval_new.py -g fc1-mnist-100-drop-50-X-sgd-mnist-X-mnist-50000@DATE fgsm --cuda
```

**APD**

```
python adv_eval_new.py -g fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE gan_exps/_mnist-wgan-gp-1000100 fgsm --gan --cuda
```

#### Running Adversarial Attacks

Finally, we use the generated adversarial examples to attack each model. The `atk_source` can be changed to the other model directories for transfer attacks.

**SGLD**

```
python adv_eval_new.py --sgld fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE fgsm 1000 --cuda
```

**MC-Dropout (p=0.5)**

```
python adv_eval_new.py --dropout fc1-mnist-100-drop-50-X-sgd-mnist-X-mnist-50000@DATE fgsm 1000 --cuda
```

**APD**

```
python adv_eval_new.py --gan fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE gan_exps/_mnist-wgan-gp-1000100 fgsm 1000 --cuda
```

**Transfer Attack**

```
python adv_eval_new.py --sgld fc1-mnist-100-X-sgld-mnist-1-1-X-mnist-50000@DATE fgsm 1000 --atk_source=fc1-mnist-100-drop-50-X-sgd-mnist-X-mnist-50000@DATE --cuda
```


## Citation

If you use this code, please cite:

```
@inproceedings{wangAPD2018,
  title={Adversarial Distillation of Bayesian Neural Network Posteriors},
  author={Kuan-Chieh Wang and Paul Vicol and James Lucas and Li Gu and Roger Grosse and Richard Zemel},
  booktitle={{International Conference on Machine Learning (ICML)}},
  year={2018}
}
```
