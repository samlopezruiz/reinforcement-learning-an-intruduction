from copy import deepcopy

import gymnasium as gym
import numpy as np
import pandas as pd


def get_dim(env: gym.Env, attr: str = 'action_space'):
    if isinstance(getattr(env, attr), gym.spaces.MultiDiscrete):
        return getattr(env, attr).nvec
    elif isinstance(getattr(env, attr), gym.spaces.Discrete):
        return getattr(env, attr).n
    else:
        raise NotImplemented


def get_ravel_dim(env: gym.Env, attr: str):
    # Maximal state or action ravel index
    n = get_dim(env, attr)
    if isinstance(n, np.ndarray):
        return np.ravel_multi_index(n - 1, n) + 1
    else:
        return n


def get_choices_ravel_ix(env: gym.Env, attr: str = 'action_space'):
    n = get_dim(env, attr)
    if isinstance(n, np.ndarray):
        return np.arange(int(np.prod(n)))
    else:
        return np.arange(n)


def get_ravel_state_idx(state, env: gym.Env):
    if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return np.ravel_multi_index(state, env.observation_space.nvec)
    else:
        return state

def get_unravel_item(ridx: int, env: gym.Env, attr: str = 'action_space'):
    if isinstance(getattr(env, attr), gym.spaces.MultiDiscrete):
        return np.array(np.unravel_index(ridx, getattr(env, attr).nvec))
    else:
        return ridx

def get_unravel_action(action_ridx: int, env: gym.Env):
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        return np.array(np.unravel_index(action_ridx, env.action_space.nvec))
    else:
        return action_ridx

def run_episode(env, policy, greedy=False, render=False):
    """ Follow policy through an episode and return arrays of visited actions, states and returns """
    ravel_actions = get_choices_ravel_ix(env, 'action_space')
    state_ridxs, action_ridxs, rewards = [], [], []

    state = env.reset()
    if render:
        env.render()

    done = False
    while not done:
        state_ridx = get_ravel_state_idx(state, env)
        state_ridxs.append(state_ridx)

        # Sample action from the policy
        if greedy:
            action_ridx = np.argmax(policy[state_ridx])
        else:
            action_ridx = np.random.choice(ravel_actions, p=policy[state_ridx])
        action = get_unravel_action(action_ridx, env)
        action_ridxs.append(action_ridx)

        # Step the environment forward and take the sampled action
        state, reward, done, _, info = env.step(action)
        rewards.append(reward)

        if render:
            env.render()

    if render:
        import matplotlib.pyplot as plt
        plt.show()

    return state_ridxs, action_ridxs, rewards


def q_to_2d_v(q, env):
    q_ = deepcopy(q)
    unvisited = np.where(q_ == 0)
    q_[unvisited] = np.nan
    q_ = pd.DataFrame(q_) #to ignore nans in max()
    v = q_.max(axis=1).values.reshape(env.observation_space.nvec)
    return v.T