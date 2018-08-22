
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np


# a mixer (linear layer)
class waspSlicer(nn.Module):
    def __init__(self, opt, ngpu=1, pstart = 0, pend=1):
        super(waspSlicer, self).__init__()
        self.ngpu = ngpu
        self.pstart = pstart
        self.pend = pend
    def forward(self, input):
        output = input[:,self.pstart:self.pend].contiguous()
        return output


# Dense block in encoder.
# Dense block in encoder.
class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

# Dense block in encoder.
class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)


# an encoder architecture
# Densely connected convolutions.
class waspDenseEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[]):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.BatchNorm2d(nc),
                nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 32 x 32
                DenseBlockEncoder(ndf, 6),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args),

                # state size. (ndf*2) x 16 x 16
                DenseBlockEncoder(ndf*2, 12),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args),

                # state size. (ndf*4) x 8 x 8
                DenseBlockEncoder(ndf*4, 24),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 4 x 4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output

class waspDenseDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0,1]):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*4),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf*4, 24),
            DenseTransitionBlockDecoder(ngf*4, ngf*2),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf*2, 12),
            DenseTransitionBlockDecoder(ngf*2, ngf),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
    def forward(self, inputs):
        return self.main(inputs)


# The encoders
class Dense_Encoders_AE_SliceSplit(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_AE_SliceSplit, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zPmixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = opt.pdim)
        self.zEmixer = waspSlicer(opt, ngpu=1, pstart = opt.pdim, pend = opt.pdim + opt.edim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zp    = self.zPmixer(self.z)
        self.ze    = self.zEmixer(self.z)
        return self.z, self.zp, self.ze


# The encoders of VAE
class Dense_Decoders_AE(nn.Module):
    def __init__(self, opt):
        super(Dense_Decoders_AE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output     = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output
