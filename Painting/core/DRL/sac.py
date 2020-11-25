import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

criterion = nn.MSELoss()

Decoder = FCN()
Decoder.load_state_dict(torch.load('../trans_model.pkl'))

def decode(x, canvas): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)
    
class SAC(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size   
        self.alpha = 0.001
        
        self.entropy_tuning = True
        if self.entropy_tuning:
            self.entropy_target = -torch.Tensor([65.]).to(device)
            self.log_alpha = torch.tensor([0.], requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=1e-3)

        self.pi = ResGaussianActor(9, 18, 65) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.q1 = ResNet_wobn(3 + 9, 18, 1) # add the last canvas for better prediction
        self.q1_target = ResNet_wobn(3 + 9, 18, 1) 
        self.q2 = ResNet_wobn(3 + 9, 18, 1)
        self.q2_target = ResNet_wobn(3 + 9, 18, 1) 

        self.pi_optim  = Adam(self.pi.parameters(), lr=1e-2)
        self.q1_optim  = Adam(self.q1.parameters(), lr=1e-2)
        self.q2_optim  = Adam(self.q2.parameters(), lr=1e-2)

        if (resume != None):
            self.load_weights(resume)

        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        
        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device()        

    def play(self, state):
        state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.max_step, coord.expand(state.shape[0], 2, 128, 128)), 1)
        pi_action, log_pi_prob = self.pi(state)
        return pi_action, log_pi_prob
    
    def evaluate(self, state, action, target=False):
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        # gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)        
        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        
        if target:
            V1 = self.q1_target(merged_state)
            V2 = self.q2_target(merged_state)
            return (self.discount * V1 + L2_reward, self.discount * V1 + L2_reward), L2_reward
        else:
            V1 = self.q1(merged_state)
            V2 = self.q2(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_V1', V1.mean(), self.log)
                self.writer.add_scalar('train/expect_V2', V2.mean(), self.log)
                self.writer.add_scalar('train/L2_reward', L2_reward.mean(), self.log)
            return (self.discount * V1 + L2_reward, self.discount * V2 + L2_reward), L2_reward
    
    def update_policy(self, lr):
        self.log += 1
        
        for param_group in self.q1_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.q2_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.pi_optim.param_groups:
            param_group['lr'] = lr[1]
            
        # Sample batch
        state, action, reward, \
            next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        # self.update_gan(next_state)
        
        with torch.no_grad():
            next_action, log_next_pi = self.play(next_state)
            tar_q_values, _ = self.evaluate(next_state, next_action, True)
            target_q = torch.min(tar_q_values[0], tar_q_values[1])
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * (target_q - self.alpha * log_next_pi)
                
        cur_q_values, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()
        
        value_loss = criterion(cur_q_values[0], target_q)
        self.q1.zero_grad()
        value_loss.backward(retain_graph=True)
        self.q1_optim.step()
        value_loss = criterion(cur_q_values[1], target_q)
        self.q2.zero_grad()
        value_loss.backward(retain_graph=True)
        self.q2_optim.step()

        action, log_pi = self.play(state)
        pre_q_values, _ = self.evaluate(state.detach(), action)
        pre_q = torch.min(pre_q_values[0], pre_q_values[1])
        policy_loss = (-pre_q + self.alpha * log_pi).mean()
        self.pi.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.pi_optim.step()
        
        
        if self.entropy_tuning:
            alpha_loss = -( self.log_alpha * (log_pi + self.entropy_target).detach() ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            self.writer.add_scalar('Loss/alpha', self.alpha, self.log)
        
        
        # Target update
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action, _ = self.play(state)
            action = to_numpy(action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.pi.load_state_dict(torch.load('{}/pi.pkl'.format(path)))
        self.q1.load_state_dict(torch.load('{}/q1.pkl'.format(path)))
        self.q2.load_state_dict(torch.load('{}/q2.pkl'.format(path)))
        #load_gan(path)
        
    def save_model(self, path):
        self.pi.cpu()
        self.q1.cpu()
        self.q2.cpu()
        torch.save(self.pi.state_dict(),'{}/pi.pkl'.format(path))
        torch.save(self.q1.state_dict(),'{}/q1.pkl'.format(path))
        torch.save(self.q2.state_dict(),'{}/q2.pkl'.format(path))
        #save_gan(path)
        self.choose_device()

    def eval(self):
        self.pi.eval()
        self.q1.eval()
        self.q1_target.eval()
        self.q2.eval()
        self.q2_target.eval()
    
    def train(self):
        self.pi.train()
        self.q1.train()
        self.q1_target.train()
        self.q2.train()
        self.q2_target.train()
    
    def choose_device(self):
        Decoder.to(device)
        self.pi.to(device)
        self.q1.to(device)
        self.q1_target.to(device)
        self.q2.to(device)
        self.q2_target.to(device)
