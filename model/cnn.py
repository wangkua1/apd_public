import pdb

import torch
import torchvision.models as tvm
import torch.nn as nn
from torch.autograd import Variable

class cnn_pytorch_base(nn.Module):
    def __init__(self):
        super(cnn_pytorch_base, self).__init__()
        self.posterior_samples = []
        self.posterior_weights = []

    def state_dict(self):
        return self.conv.state_dict(), self.fc.state_dict()

    def load_state_dict(self, sd):
        self.conv.load_state_dict(sd[0])
        self.fc.load_state_dict(sd[1])


class cnn(cnn_pytorch_base):
    """docstring for cnn"""

    def __init__(self, Hn, input_dim=[1,28,28],output_dim=10):
        super(cnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
        )
        D = input_dim[-1]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, output_dim),
        )

    def forward(self, input):
        # input = input.permute(0,3,1,2)
        # tmp = self.conv(input)
        tmp = self.conv(input)

        tmp = tmp.view(-1,self.fc_dim)
        return self.fc(tmp)

class identity_layer(nn.Module):
    def __init__(self, *args):
        super(identity_layer, self).__init__()

    def forward(self, x):
        return x

class cnn_globe(cnn_pytorch_base):
    """popular architecture... specifically following ladder net
    """
    def __init__(self, Hn=1, input_dim=[1,28,28], output_dim=10, dropout=0.2, softmax=False, bn=''):
        super(cnn_globe, self).__init__()
        hid1=int(96*Hn)
        hid2=int(192*Hn)
        hidfc=128
        self.softmax = softmax
        self.input_dim = input_dim
        self.output_dim = output_dim
        f_norms = {
            'bn':torch.nn.BatchNorm2d,
            'in':torch.nn.InstanceNorm2d,
            '':identity_layer,
        }
        f_norm = f_norms[bn]

        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Conv2d(input_dim[0], hid1, 3, padding=1),
            f_norm(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid1, hid1, 3, padding=1),
            f_norm(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid1, hid1, 3, padding=1),
            f_norm(hid1),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool2d((2,2), stride=2),
            f_norm(hid1),
            #
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(hid1, hid2, 3, padding=1),
            f_norm(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, hid2, 3, padding=1),
            f_norm(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, hid2, 3, padding=1),
            f_norm(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool2d((2,2), stride=2),
            f_norm(hid2),
            #
            torch.nn.Dropout(.3), # was 0.3 before when hitting baseline
            torch.nn.Conv2d(hid2, hid2, 3, padding=0),
            f_norm(hid2),
            torch.nn.LeakyReLU(.2),
            ## 1x1 convs on 6x6 images
            torch.nn.Conv2d(hid2, hid2, 1, padding=0),
            f_norm(hid2),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(hid2, self.output_dim, 1, padding=0),
            f_norm(self.output_dim),
            torch.nn.LeakyReLU(.2),
        )
        

    def forward(self, input):
        # pdb.set_trace()
        N = input.size()[0]
        # input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        ## (N, output_dim, 6,6) -> (N, , -1)
        tmp = tmp.view(N, self.output_dim, -1)
        ## average pool across channels
        tmp = tmp.mean(-1).view(N, self.output_dim)
        tmp = tmp.clamp(max=1e10,min=1e-10)
        return tmp

    def state_dict(self):
        return [self.conv.state_dict()]

    def load_state_dict(self, sd):
        self.conv.load_state_dict(sd[0])



# class ResNet(object):
#     def __init__(self, suffix=18):
#         super(ResNet, self).__init__()
#         self.model = eval('tvm.resnet%g'%suffix)()
#     def eval(self):
#         self.model.eval()

#     def train(self):
#         self.model.train()
#     def parameters(self):
#         return list(self.model.parameters())
#     def zero_grad(self):
#         self.model.zero_grad()
#     def forward(self, input):
#         # input = input.permute(0,3,1,2)
#         tmp = self.model(input)
#         return tmp

#     def save(self, name):
#         dic = {}
#         dic['model'] = self.model.state_dict()
#         torch.save(dic, name)

#     def load(self, name):
#         dic = torch.load(name)
#         self.model.load_state_dict(dic['model'])
#     def type(self, dtype):
#         self.model.type(dtype)
