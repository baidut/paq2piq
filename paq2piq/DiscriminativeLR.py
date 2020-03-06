import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torchvision
import functools
import torch
from typing import Union

'''
Developped by the Fastai team for the Fastai library
From the fastai library
https://www.fast.ai and https://github.com/fastai/fastai
'''

###############################################################################
#Unmodified classes and functions:

class PrePostInitMeta(type):
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        old_init = x.__init__
        def _pass(self): pass
        @functools.wraps(old_init)
        def _init(self,*args,**kwargs):
            self.__pre_init__()
            old_init(self, *args,**kwargs)
            self.__post_init__()
        x.__init__ = _init
        if not hasattr(x,'__pre_init__'):  x.__pre_init__  = _pass
        if not hasattr(x,'__post_init__'): x.__post_init__ = _pass
        return x

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self): super().__init__()
    def __init__(self): pass

class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p:nn.Parameter): self.val = p
    def forward(self, x): return x

def children(m:nn.Module):
    "Get children of `m`."
    return list(m.children())

def num_children(m:nn.Module):
    "Get number of children modules in `m`."
    return len(children(m))

def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children

def even_mults(start:float, stop:float, n:int)->np.ndarray:
    "Build log-stepped array from `start` to `stop` in `n` steps."
    mult = stop/start
    step = mult**(1/(n-1))
    return np.array([start*(step**i) for i in range(n)])

flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if num_children(m) else [m]
###############################################################################

'''
Modified version of lr_range from fastai
https://github.com/fastai/fastai/blob/master/fastai/basic_train.py#L185
'''
def lr_range(net:nn.Module, lr:slice, model_len:int)->np.ndarray:
        "Build differential learning rates from `lr`."

        if not isinstance(lr,slice): return lr
        if lr.start: res = even_mults(lr.start, lr.stop, model_len)
        else: res = [lr.stop/10]*(model_len-1) + [lr.stop]
        return res

def unfreeze_layers(model:nn.Sequential, unfreeze:bool=True)->None:
    "Unfreeze or freeze all layers"

    for layer in model.parameters():
        layer.requires_grad = unfreeze

def build_param_dicts(layers:nn.Sequential, lr:list=[0], return_len:bool=False)->Union[int,list]:
    '''
    Either return the number of layers with requires_grad is True
    or return a list of dictionnaries containing each layers on its associated LR"
    Both weight and bias are check for requires_grad is True
    '''

    params = []
    idx = 0
    for layer in layers:
        param = []
        if (hasattr(layer, "requires_grad") and layer.requires_grad):
            #To implement for custom nn.Parameter()
            print("Custom nn.Parameter() not supported")
        if(hasattr(layer, "weight") and layer.weight.requires_grad):
            param.append(layer.weight)
        if (hasattr(layer, "bias") and hasattr(layer.bias, "requires_grad") and layer.bias.requires_grad):
            param.append(layer.bias)
        if param: params.append({'params': param, 'lr': f'{lr[idx]}'}); idx += 1
        if return_len: idx = 0 #We don't want to increment idx here.

    return len(params) if return_len else params

def discriminative_lr_params(net:nn.Module, lr:slice, unfreeze:bool=True)->Union[list,np.ndarray,nn.Sequential]:
    '''
    Flatten our model and generate a list of dictionnaries to be passed to the
    optimizer.
    - If only one learning rate is passed as a slice the last layer will have the
    corresponding learning rate and all other ones will have lr/10
    - If two learning rates are passed such as slice(min_lr, max_lr) the last
    layer will have max_lr as a learning rate and the first one will have min_lr.
    All middle layers will have learning rates logarithmically interpolated
    ranging from min_lr to max_lr
    '''

    layers = nn.Sequential(*flatten_model(net)) #Flatten/ungroup our model
    if unfreeze: unfreeze_layers(layers, True)  #Unfreeze all layers

    #Return the number of layer where requires_grad is True (bias + weight)
    model_len = build_param_dicts(layers, return_len=True)

    #Create the list of learning rates
    list_lr = lr_range(net, lr, model_len)

    #Create our optimizer parameters list of dictionnaries
    params_layers = build_param_dicts(layers, list_lr)

    return params_layers, np.array(list_lr), layers
