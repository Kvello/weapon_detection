from . import  custom_lr_schedulers
import torch.optim.lr_scheduler as lr_scheduler
import logging
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Dict, Union, Tuple
from collections import deque
import matplotlib.pyplot as plt

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss,quiet=False):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if not quiet:
                print("Validation loss increased, counter: {}".format(self.counter))
            if self.counter >= self.patience:
                if not quiet:
                    print("Count: {}, Patience: {} - quitting".format(self.counter,self.patience))
                return True
        return False

def validate(model:nn.Module,
             valloader:data.DataLoader,
                loss_fn:nn.Module,
                device:str="cpu",
                ):
    r"""
    General purpose validation function
    Args:
        model (nn.Module): Model to validate
        valloader (data.DataLoader): Validation dataset
        loss_fn (nn.Module): Loss function to use
        device (str): Device to use for validation. Default: 'cpu'
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in valloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss += loss_fn(y_pred,y).item()
            pred = y_pred.argmax(dim=1,keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(valloader)
    accuracy = correct / len(valloader.dataset)
    return loss, accuracy

def train(model:nn.Module,
          dataloader:data.DataLoader,
          valloader:data.DataLoader=None,
          epochs:int=100,
          lr:Union[float,Tuple[float,lr_scheduler.LRScheduler],Tuple[float,str]]=0.001,
          momentum:float=0,
          weight_decay:float=0,
          device:str="cpu",
          loss:str="cross_entropy",
          optimizer:str="sgd",
          plot_loss:bool=False,
          quiet:bool=False,
          early_stopper:EarlyStopper=None
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
        logging.error("Loss function {} not implemented yet".format(loss))
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
                logging.error("Learning rate scheduler {} not implemented yet".format(lr))
                raise ValueError("Learning rate scheduler {} not implemented yet".format(lr))
        elif isinstance(lr,lr_scheduler.LRScheduler):
            scheduler = lr
        else:
            logging.error("Learning rate scheduler {} not implemented yet".format(lr))
            raise ValueError("Learning rate scheduler {} not implemented yet".format(lr))
    elif isinstance(lr,float):
        lr_init = lr
        scheduler = None
    else:
        logging.error("Learning rate {} not implemented yet".format(lr))
        raise ValueError("Learning rate {} not implemented yet".format(lr))
    
        
    if optimizer.lower()=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
    elif optimizer.lower()=="adam":
        optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
        if momentum != 0:
            logging.warning("Momentum is not used for Adam optimizer, ignored")
    elif optimizer.lower()=="AdamW":
        if momentum != 0:
            logging.warning("Momentum is not used for AdamW optimizer, ignored")
        optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    else:
        logging.error("Optimizer {} not implemented yet".format(optimizer))
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
        epoch_loss = 0
        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            ls = loss_fn(y_pred,y)
            epoch_loss += ls.item()
            ls.backward()
            optimizer.step()
            print("Optimzer step")
            if scheduler is not None:
                scheduler.step()
        epoch_loss /= len(dataloader)
        if not quiet:
            print("Epoch {}: Loss: {}".format(epoch,epoch_loss))
            if valloader is not None:
                val_loss, val_accuracy = validate(model,valloader,loss_fn,device)
                print("Validation loss: {}, Validation accuracy: {}".format(val_loss,val_accuracy))
        if early_stopper is not None:
            if early_stopper.early_stop(val_loss,quiet):
                print("Early stopping")
                break
        if plot_loss:
            loss_history.append(epoch_loss)
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            plt.draw()
            plt.show()
