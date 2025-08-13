import torch
from torch import nn


def safe_normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

class SPRegularization(nn.Module):
    """Code partially taken from https://github.com/omegafragger/ldifs_code/"""
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            output += torch.norm(param - self.source_weight[name]) ** 2
        return output

class LDIFSRegularization(nn.Module):
    """Code partially taken from https://github.com/omegafragger/ldifs_code/"""
    def __init__(self, source_model, target_model):
        super(LDIFSRegularization, self).__init__()
        self.source_model = source_model
        self.target_model = target_model

        self.L = 1
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad = False

    def forward(self, batch_ppi):
        self.target_model.eval()
        outs1, outs2 = [self.source_model(batch_ppi)], [self.target_model(batch_ppi)]
        self.target_model.train()
        feats1, feats2, diffs = [], [], []

        for kk in range(self.L):
            feats1.append(safe_normalize_tensor(outs1[kk]))
            feats2.append(safe_normalize_tensor(outs2[kk]))
            diff = ((feats1[kk] - feats2[kk]) ** 2)
            if (len(diff.shape) == 3):
                diff = diff.mean(dim=-1).mean(dim=-1)
            elif (len(diff.shape) == 2):
                diff = diff.mean(dim=-1)
            diffs.append(diff)
        ldifs = torch.stack(diffs, dim=1).mean(dim=-1).mean(dim=0)
        return ldifs
