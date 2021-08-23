import torch
import torch.nn as nn
import numpy as np
from pytorch_utils import build_activation
from torch_deps.weight_init import basic_weight_init, basic_weight_init_he_normal_relu, basic_weight_init_he_uniform_relu, basic_weight_init_glorut_uniform

class Tc1Net(nn.Module):

    def __init__(self,
               
                conv: list,
                dense: list,
                activation: str,
                out_activation: str,
                dropout: int,
                classes: int,
                pool: list,
                locally_connected: bool,
                input_dim: int,
                 
                 ):

        super(Tc1Net, self).__init__()

        self.__tc1_net = nn.Sequential()

        module_index = 0
        prev_dim = list(input_dim)

        dense_first = True

        layer_list = list(range(0, len(conv), 3))
        for l, i in enumerate(layer_list):
            filters = conv[i]
            filter_len = conv[i+1]
            stride = conv[i+2]
            print(i/3, filters, filter_len, stride)
            if pool:
                pool_list=pool
                if type(pool_list) != list:
                    pool_list=list(pool_list)

            if filters <= 0 or filter_len <= 0 or stride <= 0:
                    break
            dense_first = False

            if locally_connected:
                test = 1
                #    model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                #input layer
                if i == 0:
                    self.__tc1_net.add_module('conv_%d' % module_index,
                                              nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=filter_len, stride=stride, padding=0))                
                    prev_dim[0] = filters
                    prev_dim[1] = (prev_dim[1] - (filter_len - 1) -1)//stride + 1 #need to add stride
                    #model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
                else:
                    #model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
                    self.__tc1_net.add_module('conv_%d' % module_index,
                                              nn.Conv1d(in_channels=prev_dim[0], out_channels=filters, kernel_size=filter_len, stride=stride, padding=0))                
                    prev_dim[0] = filters
                    prev_dim[1] = (prev_dim[1] - (filter_len - 1) -1)//stride + 1 #need to add stride            

            #model.add(Activation(gParameters['activation']))
            self.__tc1_net.add_module('activation_%d' % module_index, 
                                      build_activation(activation))
            if pool:
                #model.add(MaxPooling1D(pool_size=pool_list[i//3]))pool_list[i//3]
                pool_size = pool_list[i//3]
                self.__tc1_net.add_module('activation_%d' % module_index,
                                          nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0))
                prev_dim[1] = (prev_dim[1] - (pool_size - 1) -1)//pool_size + 1  #need to add stride
            
            module_index += 1
        if not dense_first:
            #model.add(Flatten())
            self.__tc1_net.add_module('flatten_%d' % module_index,
                                      nn.Flatten()) 
            prev_dim[0] = prev_dim[0]*prev_dim[1]
            prev_dim[1] = 1
            module_index += 1
        
        for i, layer in enumerate(dense):
            if layer:
                if i == 0 and dense_first:
                    #model.add(Dense(layer, input_shape=(x_train_len, 1)))
                    self.__tc1_net.add_module('dense_%d' % module_index,
                                               nn.Linear(prev_dim[0], layer))
                    prev_dim[0] = layer 
                else:
                    #model.add(Dense(layer))
                    self.__tc1_net.add_module('dense_%d' % module_index,
                                               nn.Linear(prev_dim[0], layer))
                    prev_dim[0] = layer 
                #model.add(Activation(gParameters['activation']))
                self.__tc1_net.add_module('activation_%d' % module_index, 
                                          build_activation(activation))
                if dropout:
                    #model.add(Dropout(gParameters['dropout']))
                    self.__tc1_net.add_module('dropout_%d' % module_index, 
                                        nn.Dropout(p=dropout))
                module_index += 1

        # Weight Initialization ###############################################
        if activation == 'relu':
            self.__tc1_net.apply(basic_weight_init_he_uniform_relu)
        else:
            self.__tc1_net.apply(basic_weight_init)
        

        if dense_first:
            #model.add(Flatten())
            self.__tc1_net.add_module('flatten_%d' % module_index,
                                      nn.Flatten()) 
            prev_dim[0] = prev_dim[0]*prev_dim[1]
            prev_dim[1] = 1
            module_index += 1

        #model.add(Dense(gParameters['classes']))
        self.__tc1_net.add_module('dense_%d' % module_index,
                                  nn.Linear(prev_dim[0], classes))
        prev_dim[0] = classes
        module_index += 1

        #model.add(Activation(gParameters['out_activation']))
        self.__tc1_net.add_module('activation_%d' % module_index, 
                                  build_activation(out_activation, dim=1))
        


    def forward(self, x):
        return self.__tc1_net(x)

