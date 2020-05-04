import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable

class ConcreteDropout(nn.Module):
    """This module allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = nn.Sequential(ConcreteDropout(Linear_relu(1, nb_features),
        input_shape=(batch_size, 1), weight_regularizer=1e-6, dropout_regularizer=1e-5))
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = nn.Sequential(ConcreteDropout(Conv2D_relu(channels_in, channels_out),
        input_shape=(batch_size, 3, 128, 128), weight_regularizer=1e-6,
        dropout_regularizer=1e-5))
    ```
    # Arguments
        layer: a layer Module.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, input_shape, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1)).cuda()
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.01
        self.p = nn.functional.sigmoid(self.p_logit)

        # Check if batch size is the same as unif_noise, if not take care
        unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size())))).cuda()

        drop_prob = (torch.log(self.p + eps)
                    - torch.log(1 - self.p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square

class Architecture(nn.Module):

    def __init__(self, n_classes, input_shape, usegpu=True):
        super(Architecture, self).__init__()
        
        batchNorm_momentum = 0.1
        h = input_shape[2]
        w = input_shape[3]
        
        #self.conv11 = ConcreteDropout(nn.Conv2d(input_shape[1], 64, kernel_size=3, padding=1),input_shape)
        self.conv11 = nn.Conv2d(input_shape[1], 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        #self.conv12 = ConcreteDropout(nn.Conv2d(64, 64, kernel_size=3, padding=1),[input_shape[0], 64, h, w])
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        #self.conv21 = ConcreteDropout(nn.Conv2d(64, 128, kernel_size=3, padding=1),[input_shape[0], 128, h, w])
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        
        #self.conv22 = ConcreteDropout(nn.Conv2d(128, 128, kernel_size=3, padding=1),[input_shape[0], 128, h, w])
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        
        #self.conv31 = ConcreteDropout(nn.Conv2d(128, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #self.conv32 = ConcreteDropout(nn.Conv2d(256, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #self.conv33 = ConcreteDropout(nn.Conv2d(256, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #self.conv41 = ConcreteDropout(nn.Conv2d(256, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv42 = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv43 = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv51 = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv52 = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        self.conv53 = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        #self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        #self.conv53d = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv52d = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv51d = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        #self.conv43d = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)        
        
        #self.conv42d = ConcreteDropout(nn.Conv2d(512, 512, kernel_size=3, padding=1),[input_shape[0], 512, h, w])
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.conv41d = ConcreteDropout(nn.Conv2d(512, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        #self.conv33d = ConcreteDropout(nn.Conv2d(256, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #self.conv32d = ConcreteDropout(nn.Conv2d(256, 256, kernel_size=3, padding=1),[input_shape[0], 256, h, w])
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #self.conv31d = ConcreteDropout(nn.Conv2d(256,  128, kernel_size=3, padding=1),[input_shape[0], 128, h, w])
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        #self.conv22d = ConcreteDropout(nn.Conv2d(128, 128, kernel_size=3, padding=1),[input_shape[0], 128, h, w])
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        
        #self.conv21d = ConcreteDropout(nn.Conv2d(128, 64, kernel_size=3, padding=1),[input_shape[0], 64, h, w])
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = ConcreteDropout(nn.Conv2d(64, 64, kernel_size=3, padding=1),[input_shape[0], 64, h, w])
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        self.conv11d = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
    
    
    def outputSize(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)

    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
    
    def regularisation_loss(self):
        '''reg_loss = self.conv11.regularisation()
        reg_loss += self.conv12.regularisation()
        reg_loss += self.conv21.regularisation()
        reg_loss += self.conv22.regularisation()
        reg_loss += self.conv31.regularisation()
        reg_loss += self.conv32.regularisation()
        reg_loss += self.conv33.regularisation()
        reg_loss += self.conv41.regularisation()
        reg_loss += self.conv42.regularisation()
        reg_loss += self.conv43.regularisation()
        reg_loss += self.conv51.regularisation()
        reg_loss += self.conv52.regularisation()'''
        reg_loss = self.conv53.regularisation()
        '''reg_loss += self.conv53d.regularisation()
        reg_loss += self.conv52d.regularisation()
        reg_loss += self.conv51d.regularisation()
        reg_loss += self.conv43d.regularisation()
        reg_loss += self.conv42d.regularisation()
        reg_loss += self.conv41d.regularisation()
        reg_loss += self.conv33d.regularisation()
        reg_loss += self.conv32d.regularisation()
        reg_loss += self.conv31d.regularisation()
        reg_loss += self.conv22d.regularisation()
        reg_loss += self.conv21d.regularisation()
        reg_loss += self.conv12d.regularisation()'''
        reg_loss += self.conv12d.regularisation()
        return reg_loss