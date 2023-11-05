from collections import namedtuple
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

ValidationResult = namedtuple("ValidationResult",["loss","accuracy"])

def validate(model:nn.Module,
             valloader:data.DataLoader,
                loss_fn:nn.Module,
                device:str="cpu")->ValidationResult:
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

    return ValidationResult(loss,accuracy)

def generate_conf_matrix(model:nn.Module,
                         loader:data.DataLoader,
                         device:str="cpu",
                         normalize=False)->np.ndarray:
    r"""
    Generate a confusion matrix for a given model
    Args:
        model (nn.Module): Model to generate confusion matrix for
        loader (data.DataLoader): Dataset to generate confusion matrix for
        device (str): Device to use for validation. Default: 'cpu'
    """
    model.eval()
    classes = loader.dataset.classes
    matrix = np.zeros((len(classes),len(classes)))
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).item()
            y_pred = model(x)
            pred = y_pred.argmax(dim=1,keepdim=True).item()
            if not isinstance(pred,int) or pred >= len(classes):
                raise ValueError("Invalid prediction: {}".format(pred))
            if not isinstance(y,int) or y >= len(classes):
                raise ValueError("Invalid label: {}".format(y.item()))
            matrix[y,pred] += 1
    if normalize:
        matrix = matrix / matrix.sum(axis=1)

    