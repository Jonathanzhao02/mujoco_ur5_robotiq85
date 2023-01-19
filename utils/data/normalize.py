import torch
import torch.nn as nn

def get_normalizers(dataset):
    

class StdNormalizationLayer(nn.Module):
    '''
    Normalization layer for unit variance and 0 mean
    '''
    def __init__(self, data_mean, data_std):
        super(StdNormalizationLayer, self).__init__()
        self.data_mean = data_mean
        self.data_std = data_std

        self.data_std[self.data_std < torch.finfo(torch.float32).eps] = torch.finfo(torch.float32).eps
    
    def forward(self, x, fwd=True):
        if fwd:
            return (x - self.data_mean) / self.data_std
        else:
            return (x * self.data_std) + self.data_mean

class MinMaxNormalizationLayer(nn.Module):
    '''
    Normalization layer to [-1, 1]
    '''
    def __init__(self, data_min, data_max):
        super(MinMaxNormalizationLayer, self).__init__()
        self.data_min = data_min
        self.data_max = data_max
        self.data_mean_range = (data_max + data_min) / 2
        self.data_half = (data_max - data_min) / 2

        self.data_half[self.data_half < torch.finfo(torch.float32).eps] = torch.finfo(torch.float32).eps
    
    def forward(self, x, fwd=True):
        if fwd:
            return (x - self.data_mean_range) / self.data_half
        else:
            return (x * self.data_half) + self.data_mean_range
