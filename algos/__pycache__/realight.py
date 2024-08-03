import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from utils import Actor, Critic
from itertools import product
import numpy as np

class RealLight:
    
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.min_phase_time = self.args['min_phase_time']
        self.max_phase_time = self.args['max_phase_time']
        self.duration_time_step = self.args['duration_time_step']
        self.gamma = self.args['gamma']
        self.lmbda = self.args['lmbda']
        self.k_epoch = self.args['k_epoch']
        self.eps_clip = self.args['eps_clip']
        self.actor_lr = self.args['actor_lr']
        self.critic_lr = self.args['critic_lr']
        self.batch_size = self.args['batch_size']
        self.value_loss_coef = self.args['value_loss_coef']
        self.entropy_loss_coef = self.args['entropy_loss_coef']
        self.phase_lst = list(range(self.args['phase_dim']))
        self.timing_lst = list(range(self.min_phase_time, self.max_phase_time + 1, self.duration_time_step))
        self.action_lst = list(product(self.phase_lst, self.timing_lst))
        self.actor = Actor(self.args).to(self.device, **('device',))
        self.critic = Critic(self.args).to(self.device, **('device',))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr, **('lr',))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr, **('lr',))
        self.data = []
        self.actor.apply(self.init_weight_he_normal)
        self.critic.apply(self.init_weight_he_normal)

    
    def init_weight_he_normal(self, submodule):
        if isinstance(submodule, nn.Linear):
            nn.init.kaiming_normal_(submodule.weight)
            submodule.bias.data.zero_()

    
    def init_agent(self):
        self.data = []
        self.state = None
        self.action = 0
        self.prob = None
        self.reward = None
        self.next_state = None
        self.phase = 0
        self.timing = 0
        self.count = 0
        self.yellow = {
            'status': None,
            'phase': 0,
            'time': 0 }

    
    def get_action(self, state):
        state = torch.tensor(state, torch.float32, **('dtype',)).to(self.device, **('device',))
        prob = self.actor(state)
        prob = prob.detach().numpy()
        action = np.random.choice(self.args['out_dim'], 1, prob, **('p',))[0]
        (phase, timing) = self.action_lst[action]
        return (action, prob[action].item(), phase, timing)

    get_action = torch.no_grad()(get_action)
    
    def put(self, transition):
        self.data.append(transition)

    
    def make_batch(self):
        (s_lst, a_lst, prob_lst, r_lst, s_prime_lst, done_lst) = ([], [], [], [], [], [])
        for transition in self.data:
            (s, a, prob, r, s_prime, done) = transition
            s_lst.append(s)
            a_lst.append([
                a])
            prob_lst.append([
                prob])
            r_lst.append([
                r])
            s_prime_lst.append(s_prime)
            done_lst.append([
                0 if done else 1])
        s_batch = torch.tensor(s_lst, torch.float32, **('dtype',)).to(self.device, **('device',))
        a_batch = torch.tensor(a_lst, torch.int64, **('dtype',)).to(self.device, **('device',))
        prob_batch = torch.tensor(prob_lst, torch.float32, **('dtype',)).to(self.device, **('device',))
        r_batch = torch.tensor(r_lst, torch.float32, **('dtype',)).to(self.device, **('device',))
        s_prime_batch = torch.tensor(s_prime_lst, torch.float32, **('dtype',)).to(self.device, **('device',))
        done_batch = torch.tensor(done_lst, torch.float32, **('dtype',)).to(self.device, **('device',))
        return (s_batch, a_batch, prob_batch, r_batch, s_prime_batch, done_batch)

    
    def train(self):
        (s, a, prob_old, r, s_prime, done) = self.make_batch()
        r = (r - r.mean()) / (r.std() + 1e-08)
        done = done[-1]
        prob = self.actor(s, 1, **('softmax_dim',))
        curr_v = self.critic(s)
        next_v = self.critic(s_prime)
        value_old = curr_v.detach().clone()
        running_returns = 0
        previous_value = 0
        running_advantage = 0
        returns = torch.zeros_like(r)
        advantage = torch.zeros_like(r)
        for t in reversed(range(len(r))):
            running_returns = r[t][0] + self.gamma * running_returns
            running_td_error = r[t][0] + self.gamma * previous_value - curr_v.data[t]
            running_advantage = running_td_error + self.gamma * self.lmbda * running_advantage
            returns[t] = running_returns
            previous_value = curr_v.data[t]
            advantage[t] = running_advantage
        n = len(s)
        indexes = np.arange(n)
        for _ in range(self.k_epoch):
            np.random.shuffle(indexes)
            for i in range(n // self.batch_size):
                batch_index = indexes[self.batch_size * i:self.batch_size * (i + 1)]
                batch_index = torch.tensor(batch_index, torch.long, **('dtype',))
                s_samples = s[batch_index]
                a_samples = a[batch_index]
                prob_old_samples = prob_old[batch_index]
                value_old_samples = value_old[batch_index]
                td_target_samples = returns[batch_index]
                advantage_samples = advantage[batch_index]
                prob_ = self.actor(s_samples, 1, **('softmax_dim',))
                curr_v_ = self.critic(s_samples)
                log_prob = torch.log(prob_)
                action_prob = prob_.gather(1, a_samples)
                action_log_prob = torch.log(action_prob)
                action_log_old_prob = torch.log(prob_old_samples)
                ratio = (action_log_prob - action_log_old_prob).sum(1, True, **('keepdim',))
                ratio = ratio.clamp_(88, **('max',)).exp()
                surr1 = ratio * advantage_samples.detach()
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_samples.detach()
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(curr_v_, td_target_samples.detach())
                entropy_loss = -torch.sum(prob_ * log_prob, -1, **('dim',)).mean()
                loss_a = actor_loss - self.entropy_loss_coef * entropy_loss
                loss_c = critic_loss
                self.actor_optimizer.zero_grad()
                loss_a.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                loss_c.backward()
                self.critic_optimizer.step()
        self.data = []


