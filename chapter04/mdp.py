import abc
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class MDP(gym.Env, abc.ABC):
    """ Extension of OpenAI gym environment to finite, stochastic Markov Decision Process (MDP) """
    gamma = None
    terminals = []

    def transitions(self, state, action):
        """ Starting from this state and taking the action, return a matrix of transition states, vector of associated
        rewards, probability of the transitions and a flag indicating a terminal transition. """
        raise NotImplementedError()

    def step(self, action):
        """ Take action from the current state and return new state, reward and whether the episode has finished. """
        transitions, rewards, ps, terminals = self.transitions(self.state, action)

        idx = np.random.choice(ps.size, p=ps)
        self.state = transitions[idx]
        return transitions[idx], rewards[idx], terminals[idx], {}

    def render(self, mode='ansi'):
        print(self.state)

def argmax_last(arr):
    # Reverse the list/array
    reversed_arr = arr[::-1]
    # Use argmax on the reversed list/array and adjust the index
    return len(arr) - 1 - reversed_arr.index(max(reversed_arr))


def extract_policy(mdp, value, tolerance=1e-5):
    """ Extract policy from a value function of an MDP.

    For each state of the MDP, the action to be executed by the policy is set to:
        policy[state] = argmax(expected_payoff[action])
    """
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete

    policy = np.zeros(shape=mdp.observation_space.nvec, dtype=int)
    for state in np.ndindex(*mdp.observation_space.nvec):
        # Calculate the optimal action for every non-terminal state
        if state in mdp.terminals:
            policy[state] = 0
            continue

        expected_returns = []
        for action in np.arange(mdp.action_space.n):
            # Get all transitions and their associated rewards and probabilities, given state and action
            transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)

            # Returns non-terminal transitions
            returns = rewards + mdp.gamma * value[tuple(transitions.T)]
            # Returns from terminal transitions
            terminal = np.where(terminals == 1)
            if terminal[0].size > 0:
                value[tuple(transitions[terminal].T)] = rewards[terminal]
                returns[terminal] = rewards[terminal]

            # Expected returns
            expected_returns += [np.sum(ps * returns)]

        # rounded_returns = np.round(expected_returns, -int(np.log10(tolerance)))
        # policy[state] = np.argmax(expected_returns)
        # policy[state] = argmax_last(expected_returns)
        policy[state] = np.random.choice(np.where(expected_returns == max(expected_returns))[0])

    return policy


def policy_iteration(mdp, save_history=False):
    """ Find and return the optimal value function and optimal policy for the MDP.

    The function uses the following version of the policy iteration algorithm:
        0. Initialize to a random policy and zero value function.
        1. Calculate value function of the current policy.
        2. Extract new policy that is greedy w.r.t. the value function.
        3. Goto 1. unless the policy stopped changing.
    """
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete
    policies, values = [], []
    old_policy = -1
    value = np.zeros(shape=mdp.observation_space.nvec)
    policy = np.random.randint(mdp.action_space.n, size=value.shape)
    pbar = tqdm(desc='Convergence=0.0%', position=0)
    while (policy != old_policy).any():
        old_policy = np.copy(policy)
        for state in np.ndindex(*mdp.observation_space.nvec):
            # Calculate the update for value function for every non-terminal state
            if state in mdp.terminals:
                continue

            action = policy[state]
            # Get all transitions and their associated rewards and probabilities, given state and action
            transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)

            # Returns non-terminal transitions
            returns = rewards + mdp.gamma * value[tuple(transitions.T)]
            # Returns from terminal transitions
            terminal = np.where(terminals == 1)
            if terminal[0].size > 0:
                value[tuple(transitions[terminal].T)] = rewards[terminal]
                returns[terminal] = rewards[terminal]

            # Expected returns
            expected_return = np.sum(ps * returns)
            value[state] = expected_return

        # Extraction of an improved policy from the updated value function
        policy = extract_policy(mdp, value)

        if save_history:
            policies.append(np.copy(policy))
            values.append(np.copy(value))

        pbar.set_description('Convergence={:.1f}%'.format((policy == old_policy).sum() /  np.prod(value.shape) * 100))
        pbar.update(1)

    history = {'value': values, 'policy': policies}

    return (value, policy, history) if save_history else (value, policy)


def value_iteration(mdp, tolerance=1e-6, save_history=False):
    """ Find and return the optimal value function and optimal policy for the MDP.

    This function uses the following version of the value iteration algorithm:
        0. Set value function for all states to 0.
        1. For every state update the value function to:
              value[state] = optimal_reward + discount_factor * optimal_payoff
           Where optimal_payoff is the maximum of expected payoffs when taking
           every possible action from the current state.
        2. Goto 1. unless the maximal value function update is less than tolerance.
        3. Extract the final optimal policy from the optimal value function.
    """
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete

    old_value = 2 * tolerance
    value = np.zeros(shape=mdp.observation_space.nvec)
    values = []
    pbar = tqdm(desc='Convergence=0.0%', position=0)
    while (np.abs(value - old_value) > tolerance).any():

        old_value = np.copy(value)
        for state in np.ndindex(*mdp.observation_space.nvec):
            # Calculate the update for value function for every non-terminal state
            if state in mdp.terminals:
                continue

            expected_returns = []
            for action in np.arange(mdp.action_space.n):
                # Get all transitions and their associated rewards and probabilities, given state and action
                transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)

                # Returns non-terminal transitions
                returns = rewards + mdp.gamma * value[tuple(transitions.T)]
                # Returns from terminal transitions
                terminal = np.where(terminals == 1)
                if terminal[0].size > 0:
                    value[tuple(transitions[terminal].T)] = rewards[terminal]
                    returns[terminal] = rewards[terminal]

                # Expected returns
                expected_returns += [np.sum(ps * returns)]


            value[state] = np.max(expected_returns)

        if save_history:
            values.append(np.copy(value))

        pbar.set_description('Convergence={:.1f}% ({:.2f})'.format((np.abs(value - old_value) <= tolerance).sum().sum() / np.prod(value.shape) * 100, np.abs(value - old_value).sum()))
        pbar.update(1)

    # Extraction of the optimal policy from the optimal value function
    policy = extract_policy(mdp, value)
    history = {'value': values, 'policy': policy}

    return (value, policy, history) if save_history else (value, policy)
