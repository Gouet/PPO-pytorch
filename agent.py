import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0., std=0.1)
        torch.nn.init.constant_(m.bias, 0.1)

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()

class ActorCritic(torch.nn.Module):
    def __init__(self, num_action, lr, clip_param, final_step, state_shape, continuous=False, std=0.0, filename='./saved_actor/actor'):
        super(ActorCritic, self).__init__()

        self.state_shape = state_shape
        self.num_action = num_action
        self.clip_param = clip_param
        self.continuous = continuous
        self.filename = filename


        if self.continuous == False:
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(state_shape[0], 150),
                torch.nn.ReLU(),
                torch.nn.Linear(150, 120),
                torch.nn.ReLU(),
                torch.nn.Linear(120, 1)
            )
        else:
            self.critic = torch.nn.Sequential(    
                torch.nn.Linear(state_shape[0], 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )

        if self.continuous == False:
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_shape[0], 150),
                torch.nn.ReLU(),
                torch.nn.Linear(150, 120),
                torch.nn.ReLU(),
                torch.nn.Linear(120, num_action),
                torch.nn.Softmax()
            )
            self.num_action = 1
        else:
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_shape[0], 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, num_action)
            )
            
        self.log_std = torch.nn.Parameter(torch.ones(num_action).to(device) * std)

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        if self.continuous == False:
            dist = Categorical(probs=mu)
        else:
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        return dist, value

    def save_model(self):
        torch.save(self.state_dict(), self.filename)

    def load_model(self):
        self.load_state_dict(torch.load(self.filename))
        self.eval()

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Print(torch.nn.Module):
    def forward(self, x):
        print('print module:', x.size())
        return x


class ActorCriticCNN(torch.nn.Module):
    def __init__(self, num_action, lr, clip_param, final_step, state_shape, continuous=False, std=0.0):
        super(ActorCriticCNN, self).__init__()

        self.state_shape = state_shape
        self.num_action = num_action
        self.clip_param = clip_param
        self.continuous = continuous

        self.learning_rate_decay = lr
        self.initial_learning_rate_value = lr
        self.final_learning_rate_step = final_step

        self.initial_clip_param = clip_param
        self.final_clip_param_step = final_step

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(inplace=True)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU(inplace=True)
        )

        self.pi = torch.nn.Sequential(
            torch.nn.Linear(512, self.num_action),
            torch.nn.Softmax(dim=-1)
        )
        self.v = torch.nn.Linear(512, 1)
        """
        self.actor = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_action),
            torch.nn.Softmax(dim=-1)
        )
        """
        #self.num_action = 1

        self.apply(atari_initializer)
        #self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        N = x.size()[0]

        conv_out = self.conv(x).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        dist = Categorical(probs=pi_out)
        value = self.v(fc_out)

        #return pi_out, v_out
        
        return dist, value

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def decay_learning_rate(self, progress):
        decay = 1.0 - (progress)
        if decay < 0.0:
            decay = 0.0
        value = decay * self.initial_learning_rate_value 
        for g in self.optimizer.param_groups:
            g['lr'] = value
        self.learning_rate_decay = value

    def decay_clip_param(self, progress):
        decay = 1.0 - (progress)
        if decay < 0.0:
            decay = 0.0
        self.clip_param = (decay * self.initial_clip_param)
