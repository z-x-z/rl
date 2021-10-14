import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# proximal policy optimization
"""
Update:
    Critic: same as ac (td error)

    Actor: KL penalty or clipped surrogate object

Operating procedures:
    1. Collect trajectory (s, a, r) from the interaction with env.
    2. Update the critic and actor according the trajectory buffer.
        2.1 Calculate advantage estimates A_t.
        2.2 Use KL penalty or clipped surrogate object to update actor.
        2.3 Calculate gain according to the reward.
        2.4 Use mean-squared error between gain and critic estimated value function to update critic.
    3. Loop until the model converged.
"""


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.1)
            # nn.init.constant_(m.bias.data, 0.01)

def tensor_wrapper(x):
    if not isinstance(x, torch.FloatTensor):
        x = torch.FloatTensor(x)
    if len(x.shape) < 2:
        x = x.unsqueeze(1)
    return x

# Deal continues action space, use normal distribution to fit the real distribution.
# Input a state, output a action distribution.
class ActorNet(nn.Module):
    def __init__(self, N_in, N_out, N_HIDDEN_LAYERS=200):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.normal_mean = nn.Linear(N_HIDDEN_LAYERS, N_out)
        self.normal_std = nn.Linear(N_HIDDEN_LAYERS, N_out)
        initialize_weights(self)
        self.distribution = torch.distributions.normal.Normal  # 选则正太分布作为近似概率分布函数

    def forward(self, s):
        s = F.relu6(self.fc1(s))  # relu6 -> [0, 6]
        mean = 2 * F.tanh((self.normal_mean(s)))  # 由于动作的范围为[-2, 2]，所以均值范围在[-2, 2]
        std = F.softplus(self.normal_std(s))  # std应该大于始终0
        return mean, std

    def choose_action(self, s):
        """return action action_log_prob"""
        mean, std = self(s)
        m = self.distribution(mean, std)
        a = m.sample()
        return a, m.log_prob(a)


class CriticNet(nn.Module):
    def __init__(self, state_dim, N_HIDDEN1=32) -> None:
        super(CriticNet, self).__init__()
        """ self.fc1 = nn.Linear(state_dim + action_dim, N_HIDDEN1)
        self.out = nn.Linear(N_HIDDEN1, 1) """
        self.fcs = nn.Linear(state_dim, N_HIDDEN1)
        self.out = nn.Linear(N_HIDDEN1, 1)
        initialize_weights(self)

    def forward(self, s):
        V = self.out(torch.relu(self.fcs(s)))
        return V


class SimplePPO:
    def __init__(self, state_dim, action_dim, discount=0.95, batch_size=256, lr_a=1e-3, lr_c=2e-3, method_name='clip') -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.BATCH_SIZE = batch_size
        self.method_name = method_name
        self.method = {'kl_pen': {'kl_target': 0.01, 'lam': 0.5}, 'clip': {'epsilon': 0.2}}[method_name]
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.buffer_init()
        self.ACTOR_UPDATE_STEPS = 10
        self.CRITIC_UPDATE_STEPS = 10
        self.build_net()
        pass

    def build_net(self):
        self.actor = ActorNet(self.state_dim, self.action_dim)
        self.critic = CriticNet(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.lr_c)

    def choose_action(self, s):
        """return action, log_prob"""
        return self.actor.choose_action(torch.FloatTensor(s))

    def get_state_value(self, s):
        return self.critic(torch.FloatTensor(s))

    def buffer_init(self):
        self.buffer = [[], [], [], []]  # (b_s, b_a, b_r, b_log_prob)

    def buffer_push(self, s, a, r, log_prob):
        self.buffer[0].append(s)
        self.buffer[1].append(a)
        self.buffer[2].append(r)
        self.buffer[3].append(log_prob)

    def learn(self, s_):
        v_s_ = self.get_state_value(s_)
        b_s, b_a, b_r, b_old_log_prob = self.buffer
        self.buffer_init()
        b_g = []
        for r in b_r:
            v_s_ = r + self.discount * v_s_
            b_g.append(v_s_)
        b_g.reverse()
        b_s, b_a, b_r, b_old_log_prob, b_g = [tensor_wrapper(_) for _ in [b_s, b_a, b_r, b_old_log_prob, b_g]]
        b_v_s = self.critic(b_s)
        advantage_kth = b_g - b_v_s.detach()
        advantage_kth = (advantage_kth - advantage_kth.mean()) / (advantage_kth.std() + 1e-7)  # normalization
        # update actor
        for _ in range(self.ACTOR_UPDATE_STEPS):
            _, b_new_log_prob = self.actor(b_s)
            ratio = torch.exp(b_new_log_prob - b_old_log_prob)
            if self.method_name == 'kl_pen':
                a_loss = None
            elif self.method_name == "clip":
                # Calculate surrogate losses
                loss1 = ratio * advantage_kth
                loss2 = torch.clamp(ratio, 1 - self.method['epsilon'], 1 + self.method['epsilon']) * advantage_kth
                a_loss = (-torch.min(loss1, loss2)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()
            # update critic
            b_v_s = self.critic(b_s)  # Update b_v_s each iteration.
            c_loss = nn.MSELoss()(b_v_s, b_g)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            self.critic_optimizer.step()