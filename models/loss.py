import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, nneg, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.nneg = nneg
        self.temperature = temperature
    
    def forward(self, y_hat, y):
        bsz = y.size(0) // (self.nneg + 1)
        y_hat = y_hat.reshape(self.nneg + 1,bsz).T # reshape into (bsz, nneg + 1), true samples in [:,0]
        softmax_y_hat = F.log_softmax(-y_hat / self.temperature, dim=-1)
        lbls = F.one_hot(torch.zeros(bsz).to(int), self.nneg + 1).to(y_hat.device).float()
        loss = F.kl_div(softmax_y_hat, lbls, reduction='batchmean')
        return loss
