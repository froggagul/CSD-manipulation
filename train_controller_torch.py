import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy, MultiInputPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.sac.policies import Actor, ContinuousCritic
from stable_baselines3.common.utils import get_schedule_fn
import torch
import torch.nn as nn
from torch import Tensor
from stable_baselines3.common.vec_env import VecVideoRecorder
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default=None)
# TODO - add scratch argument

# Create the FetchPickAndPlace-v1 environment
env_id = "FetchPickAndPlace-v1"
skill_policy_path = argparser.parse_args().path
logdir = os.path.dirname(skill_policy_path)
video_folder = os.path.join(logdir, "videos")
best_model_save_path = os.path.join(logdir, "best_model")

checkpoint = torch.load(skill_policy_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

odim = checkpoint["g_stats"]["mean"].shape[0]
gdim = checkpoint["o_stats"]["mean"].shape[0]
skilldim = 3
actiondim = 7

class Normalizer:
    def __init__(self, mean, std, clip):
        self.mean = mean
        self.std = std
        self.clip = clip

    def normalize(self, x):
        x = (x - self.mean) / (self.std)
        x = torch.clamp(x, -self.clip, self.clip)
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

o_normalizer = Normalizer(
    checkpoint["o_stats"]["mean"],
    checkpoint["o_stats"]["std"],
    checkpoint["o_stats"]["clip"],
).to(device)
g_normalizer = Normalizer(
    checkpoint["g_stats"]["mean"],
    checkpoint["g_stats"]["std"],
    checkpoint["g_stats"]["clip"],
).to(device)

def preprocess_obs(obs):
    # Normalize the observations
    # TODO - same preprocess from tf training
    goal = obs['desired_goal'].float() # [B, 3]
    # achieved_goal = obs['achieved_goal'] # [B, 3]
    obs = obs['observation'].float() # [B, N]

    o = o_normalizer.normalize(obs)
    g = g_normalizer.normalize(goal)
    return o, g

class SkillConditionedActor(nn.Module):
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

class SkillConditionedCritic(nn.Module):
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

class SkillGenerator(nn.Module):
    def __init__(self, input_dim=28, hidden=256, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, output_dim)
        self.skill_min = -1.5
        self.skill_max = 1.5

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc_out(x)

        x = torch.tanh(x) # [-1, 1]
        x = self.skill_min + 0.5 * (self.skill_max - self.skill_min) * (x + 1)
        return x

class CustomActor(Actor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define additional layers or override layers if needed
        self.skill_generator = SkillGenerator(
            input_dim = odim + gdim,
        )
        self.skill_conditioned_actor = SkillConditionedActor(
            input_dim = odim + gdim + skilldim,
            output_dim = actiondim,
        )

    def get_action_dist_params(self, obs):
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        o, g = preprocess_obs(obs) # self.features_extractor(preprocess_obs(obs))
        skill = self.skill_generator(torch.cat([o, g], dim=1))

        mean, log_std = self.skill_conditioned_actor(torch.cat([o, skill, g], dim=1))

        return mean, log_std, {}

    def forward(self, obs, deterministic: bool = False) -> Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

class CustomCritic(ContinuousCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define additional layers or override layers if needed
        # for i in range(1, len(self.q_networks)):
            # self.q_networks[i] = # custom
        self.skill_generator = SkillGenerator(
            input_dim = odim + gdim,
            output_dim = skilldim,
        )
        self.q_networks = nn.ModuleList([
            SkillConditionedCritic(
                input_dim = odim + gdim + actiondim + skilldim,
            ) for _ in range(len(self.q_networks))
        ])

    def forward(self, obs: Tensor, actions: Tensor):
        o, g = preprocess_obs(obs) # self.features_extractor(preprocess_obs(obs))
        skill = self.skill_generator(torch.cat([o, g], dim=1))
        critic_input = torch.cat([o, skill, g, actions], dim=1)
        return tuple(q_net(critic_input) for q_net in self.q_networks)

class CustomPolicy(MultiInputPolicy):
    # def make_features_extractor(self, observation_space: gym.spaces.Dict) -> nn.Module:
    def make_actor(self, features_extractor=None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor = CustomActor(**actor_kwargs)
        # actor.skill_conditioned_actor.load_state_dict(checkpoint["actor"])
        # freeze the skill conditioned policy
        # for param in actor.skill_conditioned_actor.parameters():
        #     param.requires_grad = False

        return actor.to(self.device)

    def make_critic(self, features_extractor=None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic = CustomCritic(**critic_kwargs).to(self.device)
        for i in range(len(critic.q_networks)):
            critic.q_networks[i].load_state_dict(checkpoint["critic"])

        return critic.to(self.device)


# Vectorized environment (supports parallel processing)
env = make_vec_env(
    env_id,
    n_envs=1
)

# Hyperparameters for SAC
sac_config = {
    "policy": CustomPolicy,  # Updated to handle dict observation space
    "env": env,
    "learning_rate": 3e-4,
    "buffer_size": int(1e6),
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (16, "episode"),
    "verbose": 1,
    "gradient_steps": 4,
}

# Initialize SAC model
model = SAC(**sac_config)

# Create evaluation environment
eval_env = make_vec_env(env_id, n_envs=1)

eval_env = VecVideoRecorder(
    eval_env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,     # how many steps to record per evaluation episode
    name_prefix="eval_video"
)

# Callback for evaluation during training
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_save_path,
    log_path=logdir,
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=int(1e7), callback=eval_callback)
# Save the trained model
model.save("sac_fetch_pick_and_place")
