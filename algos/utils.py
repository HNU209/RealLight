import torch.nn.functional as F
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.in_dim = self.args['in_dim']
        self.hid_dim_1 = self.args['actor_hid_dim_1']
        self.hid_dim_2 = self.args['actor_hid_dim_2']
        self.out_dim = self.args['out_dim']
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim_1),
            nn.ReLU(),
            nn.Linear(self.hid_dim_1, self.hid_dim_2),
            nn.ReLU(),  
            nn.Linear(self.hid_dim_2, self.out_dim)
        )
    
    def forward(self, x, softmax_dim=0):
        x = self.fc(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.in_dim = self.args['in_dim']
        self.hid_dim = self.args['critic_hid_dim']
        self.out_dim = self.args['out_dim']
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1)
        )

    def forward(self, x):
        v = self.fc(x)
        return v