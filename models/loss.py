import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, nneg):
        super(InfoNCELoss, self).__init__()
        self.nneg = nneg
    
    def forward(self, y_hat, y):
        bsz = y.size(0) // (self.nneg + 1)
        y_hat = y_hat - torch.min(y_hat)
        p = torch.exp(-y_hat[:bsz])
        pneg = torch.sum(torch.exp(-y_hat[bsz:]).reshape(bsz, self.nneg), axis=-1)
        return torch.sum(-torch.log(p / (p + pneg))) / y.size(0)
