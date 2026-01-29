import torch
import torch.nn as nn


class SBP_layer(nn.Module):
    """
    Structured Bayesian Pruning layer

    #don't forget add kl to loss
    y, kl = sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, input_dim, z_channel = 4, init_logsigma2=9.0):
        super(SBP_layer, self).__init__()

        self.input_dim = input_dim
        self.z_channel = z_channel
        self.log_sigma2 = nn.Parameter(torch.Tensor(input_dim,1))
        self.mu = nn.Parameter(torch.Tensor(input_dim,1))

        self.mu.data.normal_(1.0, 0.01)
        self.log_sigma2.data.fill_(-init_logsigma2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(abs(self.mu) + 1e-7)
        self.log_alpha = torch.clamp(self.log_alpha, -10.0, 10.0)

        self.mask = (self.log_alpha < 0.0).float()
        if self.training:
            si = (self.log_sigma2).mul(0.5).exp_()
            eps = si.data.new(si.size()).normal_()
            multiplicator =  (self.mu + si*eps) * self.mask
        else:
            multiplicator =  (self.mu)*self.mask

        expend = multiplicator.expand(self.input_dim, self.z_channel)
        expend = expend.reshape(self.input_dim*self.z_channel)
        return expend*input #multiplicator*input

    def kl_reg_input(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))#*self.z_channel
        kl_loss = torch.sum(kl)
        return kl_loss
        
    def sparse_reg_input(self):
        s_ratio = torch.sum(self.mask.view(-1)==0.0).item() / self.mask.view(-1).size(0)
        s1 = torch.sum(self.mask.view(-1) == 0.0).item()

        self.mask = self.log_alpha.float()
        return s_ratio
