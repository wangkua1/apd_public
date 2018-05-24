# Adversarial Posterior Distillation (APD)

This repository contains the code used for the paper _Distilling the Posterior in Bayesian Neural Networks (ICML 2018)_.

## Requirements

* Python 3.6.3
* PyTorch 0.3.1.post2
* Tensorflow 1.4.0


## Experiments

### Toy 2D Classification



### Predictive Performance and Uncertainty

#### Classification Accuracy

MNIST fcNN1 (784-100-10)

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
python train_new.py model/config/fc1-mnist-100.yaml opt/config/sgld-mnist-1.yaml mnist-50000 --cuda
```


**APD**
```
python gan.py CE.fc1-mnist-100-X-sgld-mnist-1-X-mnist-50000@2017-12-12 opt/gan-config/gan1.yaml
```


MNIST fcNN2 (784-400-400-10)

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
python gan.py CE.fc-mnist-400-X-sgld-mnist-1-X-mnist-50000@2017-12-12 opt/gan-config/gan1.yaml
```






### Active Learning



### Adversarial Example Detection


