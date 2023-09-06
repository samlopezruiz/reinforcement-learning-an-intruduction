from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from compiled.utils import get_ravel_dim, run_episode


def monte_carlo_control_eps_soft(env: gym.Env,
                                 num_episodes: int,
                                 eps: float = 0.10,
                                 gamma: float = 0.9,
                                 alpha: Optional[float] = None,
                                 return_history: bool = False):

    """ First-visit Monte Carlo algorithm for eps-soft policies """

    history = defaultdict(list)

    # Maximal state and action ravel indices
    n_action_ridx = get_ravel_dim(env, 'action_space')
    n_state_ridx = get_ravel_dim(env, 'observation_space')

    # Initialize counters for state-action pairs to zero
    n_sa = np.zeros([n_state_ridx, n_action_ridx], dtype=int)

    # Non-Optimistic initialization of state-action values
    q = np.zeros([n_state_ridx, n_action_ridx], dtype=float)

    # Initialize policy to eps-soft greedy to random q
    policy = np.ones([n_state_ridx, n_action_ridx], dtype=float) / n_action_ridx

    pbar = tqdm(range(num_episodes), position=0)
    for episode in pbar:
        # Sample an episode and collect states, actions & returns
        state_ridxs, action_ridxs, rewards = run_episode(env, policy, greedy=False)

        returns = discount_rewards(gamma, rewards)
        first_visit_q_mc_update(q, n_sa, alpha, returns, state_ridxs, action_ridxs)

        # Update policy to be eps-soft greedy to updated q values
        update_policy_eps_soft(policy, q, eps, state_ridxs, n_action_ridx)

        if return_history:
            history['rewards'].append(rewards)
            history['q'].append(np.copy(q))
            history['policy'].append(policy)

        if episode % 100 == 0:
            pbar.set_description('Min return: {:.0f}'.format(min(np.cumsum(rewards[::-1]))))

    # Return deterministic greedy policy q values
    greedy_action_ridxs = np.argmax(q, axis=1)
    policy[:, :] = 0
    policy[np.arange(n_state_ridx), greedy_action_ridxs] = 1
    assert np.allclose(np.sum(policy, axis=1), 1)

    return (q, policy, history) if return_history else (q, policy)


def update_policy_eps_soft(policy, q, eps, state_ridxs, n_action_ridx):
    # state ravel indices where the q values are not equal for all actions
    state_ridxs_not_equal = [i for i in state_ridxs if not np.isclose(q[i], q[i][0]).all()]
    state_ridxs_equal = [i for i in state_ridxs if np.isclose(q[i], q[i][0]).all()]
    policy[state_ridxs_equal, :] = 1 / n_action_ridx

    greedy_action_ridxs = np.argmax(q[state_ridxs_not_equal, :], axis=1)
    policy[state_ridxs_not_equal, :] = eps / n_action_ridx
    policy[state_ridxs_not_equal, greedy_action_ridxs] = 1 - eps + eps / n_action_ridx
    assert np.allclose(np.sum(policy, axis=1), 1)


def rand_argmax(b, **kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)


def first_visit_q_mc_update(q, n_sa, alpha, returns, state_ridxs, action_ridxs):
    visited_s = set()
    # Update the state-action values with first-visit returns
    for s, a, g in zip(state_ridxs, action_ridxs, returns):
        if (s, a) not in visited_s:
            visited_s.add((s, a))
            if alpha is not None:
                q[s, a] += alpha * (g - q[s, a])
            else:
                # Increment state-action counter
                n_sa[s, a] += 1
                # Incremental update for state-action value
                q[s, a] += (g - q[s, a]) / n_sa[s, a]


def discount_rewards(gamma, rewards):
    G = 0.0
    returns = []
    for r in rewards[::-1]:
        G = gamma * G + r
        returns.append(G)
    returns = returns[::-1]
    return returns
