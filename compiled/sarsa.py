from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm

def sarsa(env, num_episodes, eps0=0.5, alpha=0.5, gamma=0.9, return_history=False, decaying_eps=True):
    """ On-policy Sarsa algorithm (with exploration rate decay) """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    history = defaultdict(list)

    # Number of available actions and maximal state ravel index
    n_action = env.action_space.n
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1

    # Initialization of action value function
    q = np.zeros([n_state_ridx, n_action], dtype=np.float)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state_ridx, n_action], dtype=np.float) / n_action

    for episode in tqdm(range(num_episodes), position=0):
        # Reset the environment
        state = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.random.choice(n_action, p=policy[state_ridx])

        done = False
        rewards = []
        while not done:
            # Step the environment forward and check for termination
            next_state, reward, done, truncated, info = env.step(action)
            next_state_ridx = np.ravel_multi_index(next_state, env.observation_space.nvec)
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])

            # Update q values
            q[state_ridx, action] += alpha * (reward + gamma * q[next_state_ridx, next_action] - q[state_ridx, action])

            # Extract eps-greedy policy from the updated q values
            eps = (eps0 / (episode + 1)) if decaying_eps else eps0
            policy[state_ridx, :] = eps / n_action
            policy[state_ridx, np.argmax(q[state_ridx])] = 1 - eps + eps / n_action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update
            state_ridx = next_state_ridx
            action = next_action
            rewards.append(reward)

        if return_history:
            history['rewards'].append(rewards)
            history['q'].append(np.copy(q))
            history['policy'].append(np.copy(policy))

    return (q, policy, history) if return_history else (q, policy)
