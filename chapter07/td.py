import gymnasium as gym
from tqdm import tqdm
import numpy as np


class SarsaLambda:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, lambda_val=0.9, replacing_trace=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.replacing_trace = replacing_trace
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.Q = np.zeros((num_states, num_actions))
        self.E = np.zeros((num_states, num_actions))

    def update(self, state, action, reward, next_state, next_action):
        delta = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]
        if self.replacing_trace:
            self.E[state, action] = 1
        else:
            self.E[state, action] += 1

        self.Q += self.alpha * delta * self.E
        self.E *= self.gamma * self.lambda_val

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])


def sarsa_eligibility_traces(env, episodes, max_steps, epsilon=0.1, verbose=True, **kwargs):
    agent = SarsaLambda(env.observation_space.n, env.action_space.n, **kwargs)

    for episode in tqdm(range(episodes), desc='Episode', disable=not verbose):
        state = env.reset()
        action = agent.choose_action(state, epsilon)

        for _ in range(max_steps):
            next_state, reward, done, info = env.step(action)
            next_action = agent.choose_action(next_state, epsilon)
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            if done:
                break

    return agent




class TDLambda:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.95, lambda_val=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.V = np.zeros(num_states)
        self.E = np.zeros(num_states)

    def update(self, state, reward, next_state):
        delta = reward + self.gamma * self.V[next_state] - self.V[state]
        self.E[state] += 1
        self.V += self.alpha * delta * self.E
        self.E *= self.gamma * self.lambda_val

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.V[state])


def td_on_policy_backward(env, episodes, max_steps, epsilon):
    agent = TDLambda(env.observation_space.n, env.action_space.n)

    for episode in tqdm(range(episodes), desc='Episode'):
        state = env.reset()

        for _ in range(max_steps):
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, reward, next_state)
            state = next_state
            if done:
                break

    return agent


def nstep_on_policy_return(v, done, states, rewards):
    """ Calculate un-discounted n-step on-policy return per (7.1) """
    assert len(states) - 1 == len(rewards)

    if not rewards:
        # Append value of the n-th state unless in the termination phase
        return 0 if done else v[states[0]]

    sub_return = nstep_on_policy_return(v, done, states[1:], rewards[1:])
    return rewards[0] + sub_return


def td_on_policy_prediction(env, policy, n, num_episodes, alpha=0.5, tderr=False):
    """ n-step TD algorithm for on-policy value prediction per Chapter 7.1. Value function updates are
     calculated by summing TD errors per Exercise 7.2 (tderr=True) or with (7.2) (tderr=False). """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    # Number of available actions and states
    n_state, n_action = env.observation_space.n, env.action_space.n,
    assert policy.shape == (n_state, n_action)

    # Initialization of value function
    v = np.ones([n_state], dtype=np.float) * 0.5

    history = []
    for episode in range(num_episodes):
        # Reset the environment and initialize n-step rewards and states
        state = env.reset()
        nstep_states = [state]
        nstep_rewards = []

        dv = np.zeros_like(v)

        done = False
        while nstep_rewards or not done:
            if not done:
                # Step the environment forward and check for termination
                action = np.random.choice(n_action, p=policy[state])
                state, reward, done, info = env.step(action)

                # Accumulate n-step rewards and states
                nstep_rewards.append(reward)
                nstep_states.append(state)

                # Keep accumulating until there's enough for the first n-step update
                if len(nstep_rewards) < n:
                    continue
                assert len(nstep_states) - 1 == len(nstep_rewards) == n

            # Calculate un-discounted n-step return per (7.1)
            v_target = nstep_on_policy_return(v, done, nstep_states, nstep_rewards)

            if tderr is True:
                # Accumulate TD errors over the episode while v is kept constant per Exercise 7.2
                dv[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])
            else:
                # Update value function toward the target per (7.2)
                v[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])

            # Remove the used n-step reward and states
            del nstep_rewards[0]
            del nstep_states[0]

            # Update value function with the sum of TD errors accumulated during the episode
            v += dv

        history += [np.copy(v)]
    return history


def nstep_off_policy_per_decision_return(v, done, states, rewards, isrs):
    """ Calculate un-discounted n-step per decision off-policy return per (7.13) """
    assert len(states) - 1 == len(rewards) == len(isrs)

    if not rewards:
        return 0 if done else v[states[0]]

    sub_return = nstep_off_policy_per_decision_return(v, done, states[1:], rewards[1:], isrs[1:])
    return isrs[0] * (rewards[0] + sub_return) + (1 - isrs[0]) * v[states[0]]


def td_off_policy_prediction(env, target, behavior, n, num_episodes, alpha=1e-3, simpler=True):
    """ n-step TD algorithm for off-policy value prediction per Chapter 7.4. Returns and value function updates
     are calculated using (7.1) and (7.9) (simpler=True) or (7.13) and (7.2) (simpler=False) """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Tuple

    # Number of available actions and states
    n_action = env.action_space.n
    n_state = [space.n for space in env.observation_space.spaces]
    assert target.shape == tuple(n_state + [n_action])
    assert behavior.shape == tuple(n_state + [n_action])

    # Initialization of value function
    v = np.zeros(n_state, dtype=np.float)

    history = []
    for episode in range(num_episodes):
        # Reset the environment and initialize n-step rewards and states
        state = env.reset()
        nstep_states = [state]
        nstep_rewards = []
        nstep_isrs = []

        done = False
        while nstep_rewards or not done:
            if not done:
                # Step the environment forward
                action = np.random.choice(n_action, p=behavior[state])
                state, reward, done, info = env.step(action)

                # Accumulate n-step rewards, states and importance sampling ratios
                nstep_rewards.append(reward)
                nstep_states.append(state)
                nstep_isrs.append(target[state + (action,)] / behavior[state + (action,)])

                # Keep accumulating until there's enough for the first n-step update
                if len(nstep_rewards) < n:
                    continue
                assert len(nstep_states) - 1 == len(nstep_rewards) == len(nstep_isrs) == n

            if simpler is True:
                # Calculate un-discounted n-step return per (7.1)
                v_target = nstep_on_policy_return(v, done, nstep_states, nstep_rewards)
                # Multiply n-step importance sampling ratios
                nstep_isr = np.prod(nstep_isrs)
                # Update value function toward the target per (7.9)
                v[nstep_states[0]] += alpha * nstep_isr * (v_target - v[nstep_states[0]])
            else:
                # Calculate un-discounted n-step return per (7.13)
                v_target = nstep_off_policy_per_decision_return(v, done, nstep_states, nstep_rewards, nstep_isrs)
                # Update value function toward the target per (7.2)
                v[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])

            # Remove the used n-step reward and states
            del nstep_rewards[0]
            del nstep_states[0]
            del nstep_isrs[0]

        history += [np.copy(v)]
    return history
