#!/home/charles/pyenv/rl/bin/python3
# coding: utf-8

import os
import random
import argparse
import itertools
from copy import deepcopy
import gym
import numpy as np
import torch
from torch.optim import Adam
from tensorboardX import SummaryWriter
import core
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device: %s"%device)

# A simple FIFO experience replay buffer
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.size = 0
        self.max_size = size
        self.ptr = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

    def sample_batch_sts(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict()
        for idx in idxs:
            pass


# fix all seed
def set_seed(seed, env=None, test_env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env:
        env.seed(seed)
        env.action_space.seed(seed)
    if test_env:
        test_env.seed(seed)
        test_env.action_space.seed(seed)

# SAC agent
class SoftActorCritic:
    def __init__(self, observation_space, action_space, ac_kwargs, 
                 gamma=0.99, alpha=0.2, lr=1e-3, polyak=0.995):
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.polyak = polyak

        self.ac = core.MLPActorCritic(observation_space, action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.ac.to(device)
        self.ac_targ.to(device)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        
        self.dynam = core.MLPModel(observation_space.shape[0], action_space.shape[0])
        self.dynam.to(device)

        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nInitial parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.m_optimizer = Adam(self.dynam.parameters(), lr=self.lr*1.0)
        
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2)
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy(), 
                      Q1Loss=loss_q1.detach().cpu().numpy(), Q2Loss=loss_q2.detach().cpu().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        a_pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, a_pi)
        q2_pi = self.ac.q2(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def compute_loss_m(self, data):
        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2']
        o2_hat, r_hat = self.dynam(o, a)
        
        loss_m_o2 = ((o2_hat - o2)**2).mean()
        loss_m_r = ((r_hat - r)**2).mean()
        loss_m = 10*loss_m_o2 + loss_m_r
        
        m_info = dict(MsLoss=loss_m_o2.detach().cpu().numpy(), MrLoss=loss_m_r.detach().cpu().numpy())
        
        return loss_m, m_info
    
    def update_model(self, data):
        self.m_optimizer.zero_grad()
        loss_m, m_info = self.compute_loss_m(data)
        loss_m.backward()
        self.m_optimizer.step()
        return m_info
        
    def update_model_based(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        self.q_optimizer.zero_grad()
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)
        
        with torch.no_grad():
            a_pi2, _ = self.ac.pi(o2)
            o_hat3, r_hat2 = self.dynam(o2, a_pi2)
            a_pi3, _ = self.ac.pi(o_hat3)
            o_hat4, r_hat3 = self.dynam(o_hat3, a_pi3)
            a_pi4, logp_a4 = self.ac.pi(o_hat4)
            
            q1_pi_targ = self.ac_targ.q1(o_hat4, a_pi4)
            q2_pi_targ = self.ac_targ.q2(o_hat4, a_pi4)
            q_pi_targ = r_hat2 + self.gamma * ( r_hat3 + self.gamma * (torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a4) )
            
            backup = r + self.gamma * (1 - d) * (q_pi_targ)
            
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        loss_q.backward()
        self.q_optimizer.step()
    
    def update(self, data):
        # update critic
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # update actor (fix critic)
        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        # update target networks
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return loss_q.detach().cpu().numpy(), q_info, loss_pi.detach().cpu().numpy(), pi_info
                
    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic)

    def save(self, record_dir, step=""):
        torch.save(self.ac.pi.state_dict(), record_dir+"./"+step+"actor.pth")
        torch.save(self.ac.q1.state_dict(), record_dir+"./"+step+"q_net_1.pth")
        torch.save(self.ac.q2.state_dict(), record_dir+"./"+step+"q_net_2.pth")
        torch.save(self.dynam.state_dict(), record_dir+"./"+step+"model.pth")


def main(env_fn, args, steps_per_epoch = 4000, epochs = 2500,
         max_ep_len = 1000, start_steps = 50000, batch_size = 100,
         update_every = 50, update_after = 50000, logdir="./"):

    writer = SummaryWriter(logdir + 'exp_sac_m')
    
    log_file = open(logdir+"train.log",'w')
    os.makedirs(logdir+"model/")

    env, test_env = env_fn(), env_fn()
    set_seed(args.seed, env=env, test_env=test_env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    sac = SoftActorCritic(env.observation_space, env.action_space,
                          ac_kwargs=dict(hidden_sizes=[args.hidden]*args.layer),
                          gamma=args.gamma, alpha=args.alpha, lr=args.lr, polyak=args.polyak)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.replay_size)

    def test_agent(num_test_episodes = 10):
        for j in range(num_test_episodes):
            ep_ret, ep_len = 0, 0
            o, d = test_env.reset(), False
            while not(d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(sac.get_action(o, True))
                ep_ret += r
                ep_len += 1
            print('TestEpRet:', ep_ret, ', TestEpLen:',ep_len)
            log_file.write("TestEpReward: {}, TestEpLen: {}\n".format(ep_ret, ep_len))

    total_steps = steps_per_epoch * epochs
    print("Total steps: %d"%total_steps)

    ep_ret, ep_len = 0, 0
    o = env.reset()
    for t in range(total_steps):
        if t > start_steps:
            a = sac.get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)

        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)

        ep_ret += r
        ep_len += 1
        o = o2

        if d or (ep_len == max_ep_len):
            print("Step:",t,', EpRet:', ep_ret, ', EpLen:', ep_len)
            log_file.write("Step: {}, EpReward: {}, EpLen: {}\n".format(t, ep_ret, ep_len))
            o = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            avg_q1_loss = 0.0
            avg_q2_loss = 0.0
            avg_pi_loss = 0.0
            avg_log_pi = 0.0
            #avg_q1 = 0.0
            #avg_q2 = 0.0
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_q, q_info, loss_pi, pi_info = sac.update(data=batch)
                avg_q1_loss += q_info['Q1Loss']
                avg_q2_loss += q_info['Q2Loss']
                avg_pi_loss += loss_pi
                avg_log_pi += pi_info['LogPi']
                #avg_q1 += q_info['Q1Vals']
                #avg_q2 += q_info['Q2Vals']
                batch = replay_buffer.sample_batch(batch_size)
                sac.update_model_based(data=batch)
            
            #writer.add_scalar('Value/q1', avg_q1/update_every, global_step=t)
            #writer.add_scalar('Value/q2', avg_q2/update_every, global_step=t)
            writer.add_scalar('Loss/q1_loss', avg_q1_loss/update_every, global_step=t)
            writer.add_scalar('Loss/q2_loss', avg_q2_loss/update_every, global_step=t)
            writer.add_scalar('Loss/pi_loss', avg_pi_loss/update_every, global_step=t)
            writer.add_scalar('Loss/pi_entropy', avg_pi_loss/update_every, global_step=t)
                
        if t >= 1000:
            batch = replay_buffer.sample_batch(batch_size)
            m_info = sac.update_model(data=batch)
            writer.add_scalar('Loss/model_state_loss', m_info['MsLoss'], global_step=t)
            writer.add_scalar('Loss/model_reward_loss', m_info['MrLoss'], global_step=t)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            test_agent()

    sac.save(logdir+"model/")
    log_file.close()
#)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--ep_lens', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--epochs', '-ep', type=int, default=2500)
    parser.add_argument('--epochs_lens', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='sac_m3')
    args = parser.parse_args()

    from datetime import datetime
    dir_path = args.exp_name + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") + '/'
    os.makedirs(dir_path)

    import yaml
    file_config = open(dir_path + "config.yaml", "w")
    yaml.dump(vars(args), file_config)
    file_config.close()
    
    main(lambda : gym.make(args.env), args, 
         steps_per_epoch=args.epochs_lens, epochs=args.epochs,
         logdir=dir_path)
