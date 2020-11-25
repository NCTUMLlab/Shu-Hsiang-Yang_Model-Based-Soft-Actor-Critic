import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP builder
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)
        self.o_head = nn.Linear(hidden_sizes[-1], obs_dim)
        self.r_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, act):
        net_out = self.net(torch.cat([obs, act], dim=-1))
        o2_hat = self.o_head(net_out)
        r_hat = self.r_head(net_out)
        return o2_hat, torch.squeeze(r_hat, -1)
    
    
class AEModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=256, z_dim=10, activation=nn.ReLU):
        super().__init__()
        self.l1 = nn.Linear([obs_dim + act_dim], hidden_sizes)
        self.l2 = nn.Linear(hidden_sizes, z_dim)
        self.l3 = nn.Linear(z_dim, hidden_sizes)        
        self.o_head = nn.Linear(hidden_sizes, obs_dim)
        self.r_head = nn.Linear(hidden_sizes, 1)

    def forward(self, obs, act):
        x = F.relu(self.l1(torch.cat([obs, act], dim=-1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        obs2_hat = self.o_head(x)
        rew_hat = self.r_head(x)
        return obs2_hat, torch.squeeze(rew_hat, -1)
    
    
class VAEModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=256, z_dim=10, activation=nn.ReLU):
        super().__init__()
        self.l1 = nn.Linear([obs_dim + act_dim], hidden_sizes)
        self.mean_head = nn.Linear(hidden_sizes, z_dim)
        self.logstd_head = nn.Linear(hidden_sizes, z_dim)
        
        self.l2 = nn.Linear(z_dim, hidden_sizes)      
        self.o_head = nn.Linear(hidden_sizes, obs_dim)
        self.r_head = nn.Linear(hidden_sizes, 1)

    def forward(self, obs, act):
        x = F.relu(self.l1(torch.cat([obs, act], dim=-1)))
        mu = F.relu(self.mean_head(x))
        logstd = F.relu(self.logstd_head(x))
        
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        x = mu + std*eps
        
        x = F.relu(self.l2(x))
        obs2_hat = self.o_head(x)
        rew_hat = self.r_head(x)
        return obs2_hat, torch.squeeze(rew_hat, -1)

#class LSTMModel(nn.Module):
#    def __init__(self, obs_dim, act_dim):
#        super().__init__()
#        self.s_embeder = nn.Linear(obs_dim, 128)
#        self.a_embeder = nn.Linear(act_dim, 128)
#        self.lstm = nn.LSTM(in_size, 256, batch_first = True)
#        self.fc_s = nn.Linear(256, obs_dim)
#        self.fc_r = nn.Linear(256, 1)

#    def forward(self, obs, act):
#        x_s = self.s_embeder(obs)
#        x_a = self.a_embeder(act)
#        x = torch.cat((x_s, x_a))
#        x, hidden = self.lstm(x, hidden)
#        x = self.fc(x)
#        return x, hidden
            
        
