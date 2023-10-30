import custom_lr_schedulers
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Dict, Union, Tuple
from collections import deque
import matplotlib.pyplot as plt

def train(model:nn.Module,
          dataloader:data.DataLoader,
          epochs:int=100,
          lr:Union[float,Tuple[float,lr_scheduler.LRScheduler],Tuple[float,str]]=0.001,
          momentum:float=0,
          weight_decay:float=0,
          device:str="cpu",
          loss:str="cross_entropy",
          optimizer:str="sgd",
          plot_loss:bool=False,
          quiet:bool=False,
          ):
    r"""
    General purpose training function
    Args:
        model (nn.Module): Model to train
        dataset (data.Dataset): Dataset to train on
        batch_size (int): Batch size to use. Default: 32
        epochs (int): Number of epochs to train for. Default: 100
        lr Union[float,Tuple[float,Callable[[int],float]],Tuple[float,str]]: Learning rate to use. 
        For adaptive learning rates, pass a tuple (init,func) where init is 
        the initial learning rate and func is a function that takes the 
        current epoch as input and returns the learning rate, or the 
        name of that funciton as a string. Default: 0.001
        momentum (float): Momentum to use for SGD. Default: 0
        weight_decay (float): Weight decay to use for SGD. Default: 0
        device (str): Device to use for training. Default: 'cpu'
        loss (str): Loss function to use - One of: 'cross_entropy', 'mse', 'nll'. Default: 'cross_entropy'
        optimizer (str): Optimizer to use - One of: 'sgd', 'adam'. Default: 'sgd'
        plot_loss (bool): Whether or not to plot the training loss. Default: False
        quiet (bool): Whether or not to print the training loss. Default: False
    """
    #TODO: Implement stacked lr schedulers
    if loss.lower()=="cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif loss.lower()=="mse":
        loss_fn = nn.MSELoss()
    elif loss.lower()=="nll":
        loss_fn = nn.NLLLoss()
    else:
        raise ValueError("Loss function {} not implemented yet".format(loss))
    if isinstance(lr,Tuple):
        lr_init, lr_func = lr
        if isinstance(lr_func,str):
            if hasattr(lr_scheduler,lr):
                scheduler = getattr(lr_scheduler,lr)
            elif hasattr(custom_lr_schedulers,lr):
                scheduler = getattr(lr_scheduler,lr)
                assert isinstance(lr,lr_scheduler.LRScheduler,
                        "Custom learning rate scheduler must be a subclass of torch.optim.lr_scheduler.LRScheduler")
            else:
                raise ValueError("Learning rate scheduler {} not implemented yet".format(lr))
        elif isinstance(lr,lr_scheduler.LRScheduler):
            scheduler = lr
        else:
            raise ValueError("Learning rate scheduler {} not implemented yet".format(lr))
    elif isinstance(lr,float):
        lr_init = lr
        scheduler = None
    else:
        raise ValueError("Learning rate {} not implemented yet".format(lr))
    
        
    if optimizer.lower()=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
    elif optimizer.lower()=="adam":
        optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer {} not implemented yet".format(optimizer))
    
    if scheduler is not None:
        scheduler = scheduler(optimizer)
    if plot_loss:
        loss_history = deque(maxlen=max(epochs,100))
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], loss_history)
    for epoch in tqdm(range(epochs)):
        model.train()
        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if not quiet:
            print("Epoch {}: Loss: {}".format(epoch,loss.item()))
        if plot_loss:
            loss_history.append(loss.item())
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            plt.draw()
            plt.show()
