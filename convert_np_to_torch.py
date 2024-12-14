import os
import torch
import torch.nn as nn
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)

path = parser.parse_args().path

save_dir = os.path.dirname(path)
np_path = os.path.join(save_dir, 'torch_data.npy')
torch_path = os.path.join(save_dir, 'torch_policy.pth')

with open(np_path, "rb") as f:
    data = np.load(f, allow_pickle=True).item()
    o_mean = data["o_mean"]
    o_std = data["o_std"]
    g_mean = data["g_mean"]
    g_std = data["g_std"]
    param_values = data["param_values"]
    o = data["o"]
    z = data["z"]
    g = data["g"]
    mu = data["mu"]
    q = data["q"]
    norm_clip = data["norm_clip"]
    norm_eps = data["norm_eps"]
    max_u = data["max_u"]

    dimo = o.shape[1]
    dimz = z.shape[1]
    dimg = g.shape[1]
    dimu = mu.shape[1]


class Normalizer:
    def __init__(self, mean, std, clip, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.clip = clip

    def normalize(self, x):
        x = (x - self.mean) / (self.std)
        x = torch.clamp(x, -self.clip, self.clip)
        return x

class Actor(nn.Module):
    def __init__(self, input_dim=31, hidden=256, output_dim=4, max_u=1.0):
        super().__init__()
        self.max_u = max_u
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, output_dim)
        self.log_std_out = nn.Linear(hidden, output_dim)
        self.log_std_min = -5
        self.log_std_max = 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # DDPG final layer typically uses tanh to bound actions
        mu = torch.tanh(self.fc_out(x)) * self.max_u

        log_std = torch.tanh(self.log_std_out(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mu, log_std

class Critic(nn.Module):
    def __init__(self, input_dim=35, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

class ActorCriticTorch(nn.Module):
    def __init__(self, dimo, dimz, dimg, dimu, max_u=1.0, hidden=256):
        super().__init__()
        self.dimo = dimo
        self.dimz = dimz
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u

        # Input dims:
        # Actor input: o + z + g = dimo + dimz + dimg = 25 + 3 + 3 = 31
        # Critic input: o + z + g + action = 31 + 4 = 35
        self.actor = Actor(input_dim=(dimo+dimz+dimg), hidden=hidden, output_dim=dimu, max_u=max_u)
        self.critic = Critic(input_dim=(dimo+dimz+dimg+dimu), hidden=hidden)

    def load_from_tf(self, param_values):
        # Actor mapping
        # fc1 weight/bias
# def mlp_gaussian_policy(x, act_dim, hidden, layers):
#     net = nn(x, [hidden] * (layers+1))
#     mu = tf.compat.v1.layers.dense(net, act_dim, activation=None)


        self.actor.fc1.weight.data = torch.from_numpy(param_values['ddpg/main/pi/_0/kernel:0'].T)
        self.actor.fc1.bias.data = torch.from_numpy(param_values['ddpg/main/pi/_0/bias:0'])

        self.actor.fc2.weight.data = torch.from_numpy(param_values['ddpg/main/pi/_1/kernel:0'].T)
        self.actor.fc2.bias.data = torch.from_numpy(param_values['ddpg/main/pi/_1/bias:0'])

        self.actor.fc3.weight.data = torch.from_numpy(param_values['ddpg/main/pi/_2/kernel:0'].T)
        self.actor.fc3.bias.data = torch.from_numpy(param_values['ddpg/main/pi/_2/bias:0'])

        self.actor.fc_out.weight.data = torch.from_numpy(param_values['ddpg/main/pi/dense/kernel:0'].T)
        self.actor.fc_out.bias.data = torch.from_numpy(param_values['ddpg/main/pi/dense/bias:0'])

        self.actor.log_std_out.weight.data = torch.from_numpy(param_values['ddpg/main/pi/dense_1/kernel:0'].T)
        self.actor.log_std_out.bias.data = torch.from_numpy(param_values['ddpg/main/pi/dense_1/bias:0'])

        # Critic mapping
        self.critic.fc1.weight.data = torch.from_numpy(param_values['ddpg/main/Q/_0/kernel:0'].T)
        self.critic.fc1.bias.data = torch.from_numpy(param_values['ddpg/main/Q/_0/bias:0'])

        self.critic.fc2.weight.data = torch.from_numpy(param_values['ddpg/main/Q/_1/kernel:0'].T)
        self.critic.fc2.bias.data = torch.from_numpy(param_values['ddpg/main/Q/_1/bias:0'])

        self.critic.fc_out.weight.data = torch.from_numpy(param_values['ddpg/main/Q/_2/kernel:0'].T)
        self.critic.fc_out.bias.data = torch.from_numpy(param_values['ddpg/main/Q/_2/bias:0'])

model = ActorCriticTorch(dimo=dimo, dimz=dimz, dimg=dimg, dimu=dimu, max_u=max_u, hidden=256)

model.load_from_tf(param_values)

o_t = torch.from_numpy(o).float()
z_t = torch.from_numpy(z).float()
g_t = torch.from_numpy(g).float()

o_mean_t = torch.from_numpy(o_mean).float()
o_std_t = torch.from_numpy(o_std).float()
g_mean_t = torch.from_numpy(g_mean).float()
g_std_t = torch.from_numpy(g_std).float()

o_normalizer = Normalizer(mean=o_mean_t, std=o_std_t, clip=norm_clip, eps=norm_eps)
g_normalizer = Normalizer(mean=g_mean_t, std=g_std_t, clip=norm_clip, eps=norm_eps)

o_t = o_normalizer.normalize(o_t)
g_t = g_normalizer.normalize(g_t)

with torch.no_grad():
    # Compute actor output
    actor_inp = torch.cat([o_t, z_t, g_t], dim=1) # shape [2, 31]
    mu_pred, std_pred = model.actor(actor_inp)

    # Compute Q using the predicted actions
    critic_inp = torch.cat([o_t, z_t, g_t, mu_pred], dim=1) # shape [2, 35]
    q_pred = model.critic(critic_inp)

print(mu, mu_pred)
print(q, q_pred)

assert np.allclose(mu, mu_pred.numpy(), atol=1e-6)
assert np.allclose(q, q_pred.numpy(), atol=1e-1)

torch.save({
    "critic": model.critic.state_dict(),
    "actor": model.actor.state_dict(),
    "o_stats": {
        "mean": o_mean_t,
        "std": o_std_t,
        "clip": norm_clip,
        "eps": norm_eps
    },
    "g_stats": {
        "mean": g_mean_t,
        "std": g_std_t,
        "clip": norm_clip,
        "eps": norm_eps
    }
}, torch_path)
