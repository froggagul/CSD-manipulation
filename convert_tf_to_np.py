import os
import pickle
import tensorflow as tf
import numpy as np
import argparse
# import torch

tf.compat.v1.disable_eager_execution()

# path as argument
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)

path = parser.parse_args().path
# path = "logs/Exp/sd000_1734002194_FetchPickAndPlace-v1_ns3_sn0_dr1_in1_sk500.0_et0.02/policy_1000.pkl"
save_dir = os.path.dirname(path)
np_path = os.path.join(save_dir, 'torch_data.npy')
torch_path = os.path.join(save_dir, 'torch_policy.pth')

with open(path, 'rb') as f: # Load the policy weights
    policy = pickle.load(f)

vars = tf.compat.v1.trainable_variables()

# Extract their current values as NumPy arrays
param_values = {}
for v in vars:
    param_values[v.name] = policy.sess.run(v)

o_mean = policy.sess.run(policy.o_stats.mean)
o_std = policy.sess.run(policy.o_stats.std)
g_mean = policy.sess.run(policy.g_stats.mean)
g_std = policy.sess.run(policy.g_stats.std)


# Now param_values is a dict mapping variable names to numpy arrays.
# You can print them or save them for use with PyTorch.
for name, val in param_values.items():
    print(name, val.shape)

# input dummy
batch_size = 2

o = np.random.randn(batch_size, policy.dimo)  # observations
z = np.random.randn(batch_size, policy.dimz)  # z input
ag = np.random.randn(batch_size, policy.dimg) # achieved goals
g = np.random.randn(batch_size, policy.dimg)  # desired goals

noise_eps = .0    # noise scale
random_eps = .0   # epsilon for random actions
use_target_net = False
compute_Q = True
exploit = False

mu, q = policy.get_actions(
    o=o,
    z=z,
    ag=ag,
    g=g,
    noise_eps=noise_eps,
    random_eps=random_eps,
    use_target_net=use_target_net,
    compute_Q=compute_Q,
    exploit=exploit
)

with open(np_path, "wb") as f:
    np.save(f, {
        "o_mean": o_mean,
        "o_std": o_std,
        "g_mean": g_mean,
        "g_std": g_std,
        "param_values": param_values,
        "o": o,
        "z": z,
        "g": g,
        "mu": mu,
        "q": q,
        "norm_clip": policy.norm_clip,
        "norm_eps": policy.norm_eps,
        "max_u": policy.max_u,
    })

print("Saved to", np_path)
print("stats", o_mean, o_std, g_mean, g_std)
