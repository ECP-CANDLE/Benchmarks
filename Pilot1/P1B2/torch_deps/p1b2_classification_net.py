import torch
import torch.nn as nn
import numpy as np
from pytorch_utils import build_activation
from torch_deps.weight_init import basic_weight_init, basic_weight_init_he_normal_relu, basic_weight_init_he_uniform_relu, basic_weight_init_glorut_uniform

class P1B2Net(nn.Module):

    def __init__(self,
                 
                layers: list,
                activation: str,
                out_activation: str,
                dropout: int,
                classes: int,
                input_dim: int,
                 ):

        super(P1B2Net, self).__init__()

        self.__p1b2_net = nn.Sequential()

        module_index = 0
        prev_dim = list(input_dim)
        
        # Define MLP architecture

        if layers is not None:
                if type(layers) != list:
                    layers = list(layers)
                for i, layer in enumerate(layers):
                    if i == 0:
                        #model.add(Dense(layer, input_shape=(x_train_len, 1)))
                        self.__p1b2_net.add_module('dense_%d' % module_index,
                                                nn.Linear(prev_dim[0], layer, True))
                        prev_dim[0] = layer 

                    else:
                        self.__p1b2_net.add_module('dense_%d' % module_index,
                                                nn.Linear(prev_dim[0], layer, True))
                        prev_dim[0] = layer 
                        
                    self.__p1b2_net.add_module('activation_%d' % module_index, 
                                            build_activation(activation))
                    if dropout:
                        #x = Dropout(gParameters['dropout'])(x)
                        self.__p1b2_net.add_module('dropout_%d' % module_index, 
                                            nn.Dropout(p=dropout))
                    module_index += 1

                #output = Dense(output_dim, activation=activation,
                #            kernel_initializer=initializer_weights,
                #            bias_initializer=initializer_bias)(x)
                
                #model.add(Dense(gParameters['classes']))
                self.__p1b2_net.add_module('dense_%d' % module_index,
                                        nn.Linear(prev_dim[0], classes))
                prev_dim[0] = classes
                module_index += 1

                #model.add(Activation(gParameters['out_activation']))
                self.__p1b2_net.add_module('activation_%d' % module_index, 
                                        build_activation(out_activation, dim=1))
        else:
            #output = Dense(output_dim, activation=activation,
            #            kernel_initializer=initializer_weights,
            #            bias_initializer=initializer_bias)(input_vector)
            self.__p1b2_net.add_module('dense_%d' % module_index,
                                    nn.Linear(prev_dim[0], classes))
            prev_dim[0] = classes
            module_index += 1

            #model.add(Activation(gParameters['out_activation']))
            self.__p1b2_net.add_module('activation_%d' % module_index, 
                                    build_activation(out_activation, dim=1))

     

        #kernel_initializer=initializer_weights,
        #    bias_initializer=initializer_bias,
        #    kernel_regularizer=l2(gParameters['reg_l2']),
        #    activity_regularizer=l2(gParameters['reg_l2']


        # Weight Initialization ###############################################
        if activation == 'relu':
            self.__p1b2_net.apply(basic_weight_init_he_uniform_relu)
        else:
            self.__p1b2_net.apply(basic_weight_init_glorut_uniform)
        


    def forward(self, x):
        return self.__p1b2_net(x)

