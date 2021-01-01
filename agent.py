import numpy as np
import torch
import torch.nn as nn


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.float), device)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        n_neurons = 256
        self.fc = nn.Sequential(
            nn.Linear(state_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU())
        self.mu = nn.Linear(n_neurons, action_dim)
        self.log_std = nn.Linear(n_neurons, action_dim)

    def forward(self, s, mean=False, with_prob=True):
        x = self.fc(s)
        mu = self.mu(x)
        if mean:
            return torch.tanh(mu)
        else:
            std = torch.clamp(self.log_std(x), -20, 2).exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()
            real_action = torch.tanh(action)
            if with_prob:
                log_prob = dist.log_prob(action).sum(1, keepdim=True)
                real_log_prob = log_prob - 2 * (np.log(2) - action - nn.Softplus()(-2*action)).sum(1, keepdim=True)
                return real_action, real_log_prob
            else:
                return real_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_neurons = 256
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 1))

    def forward(self, s, a):
        q = self.fc(torch.cat((s, a), dim=1))
        return q


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, next_sta, act, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SacAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.actor = opt_cuda(Actor(self.state_dim, self.action_dim), self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.critic2 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.target_critic1 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.target_critic2 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        soft_update(self.target_critic1, self.critic1, 1)
        soft_update(self.target_critic2, self.critic2, 1)
        self.optim_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.optim_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.log_temp = opt_cuda(torch.tensor(0.0), self.device)
        self.log_temp.requires_grad = True
        self.optim_log_temp = torch.optim.Adam([self.log_temp], lr=3e-4)
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, 1000000)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256

    def act(self, state, mean=False):
        with torch.no_grad():
            state = np_to_tensor(state, self.device).reshape(1, -1)
            action = self.actor(state, mean=mean, with_prob=False)
        return action.cpu().squeeze().numpy()

    def remember(self, state, next_state, action, reward, done):
        self.buffer.store(state.reshape(1, -1), next_state.reshape(1, -1), action, reward, 1 - done)

    def train(self):
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        si = np_to_tensor(batch['sta1'], self.device)
        sn = np_to_tensor(batch['sta2'], self.device)
        ai = np_to_tensor(batch['acts'], self.device)
        ri = np_to_tensor(batch['rews'], self.device)
        di = np_to_tensor(batch['done'], self.device)

        with torch.no_grad():
            a_next, log_p_next = self.actor(sn)
            entropy_next = - self.log_temp.exp() * log_p_next
            min_q = torch.min(self.target_critic1(sn, a_next), self.target_critic2(sn, a_next))
            yi = ri + self.gamma * di * (min_q + entropy_next)
        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        lc = ((self.critic1(si, ai) - yi) ** 2 + (self.critic2(si, ai) - yi) ** 2).mean()
        lc.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()

        self.optim_actor.zero_grad()
        a, log_p = self.actor(si)
        entropy = - self.log_temp.exp() * log_p
        la = - ((self.critic1(si, a) + self.critic2(si, a)) / 2 + entropy).mean()
        la.backward()
        self.optim_actor.step()

        self.optim_log_temp.zero_grad()
        lt = self.log_temp.exp() * (- log_p + self.action_dim).detach().mean()
        lt.backward()
        self.optim_log_temp.step()

        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
