import numpy as np
from tqdm import tqdm
from collections import defaultdict


def dyna_q(env,
           alpha=0.5,
           gamma=0.95,
           epsilon=0.1,
           n_episodes=5000,
           n_planning_steps=5,
           kappa=0.001,
           plus=False,
           verbose=True):

    # Initialize Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    tau = np.zeros([env.observation_space.n, env.action_space.n])
    history = [0]

    # Initialize model
    model = {}

    for i_episode in tqdm(range(n_episodes), disable=not verbose):
        s, _ = env.reset()
        done = False

        while not done:
            # Choose action a from state s using policy derived from Q (e.g., ε-greedy)
            if np.random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()  # explore
            else:
                a = np.argmax(Q[s, :])  # exploit

            # Take action a and observe resultant reward, and next state
            s_prime, r, done, _, _ = env.step(a)

            # Save reward in history
            history += [r]

            # Update the last-visited time step for the current state-action pair
            # tau[s, a] = i_episode
            tau += 1
            tau[s, a] = 0

            # Calculate the time-dependent exploration bonus
            # exploration_bonus = kappa * np.sqrt(i_episode - tau[(s, a)]) if plus else 0
            exploration_bonus = 0

            # Direct RL step
            Q[s, a] += + alpha * (r + exploration_bonus + gamma * np.max(Q[s_prime, :]) - Q[s, a])

            # model-learning step
            model[(s, a)] = (s_prime, r)

            # Planning Step
            for _ in range(n_planning_steps):
                # Random previously observed state and action
                s_model, a_model = list(model.keys())[np.random.choice(len(model))]
                s_prime_model, r_model = model[(s_model, a_model)]
                exploration_bonus = kappa * np.sqrt(tau[s_model, a_model]) if plus else 0

                Q[s_model, a_model] += alpha * (r_model + exploration_bonus + gamma * np.max(Q[s_prime_model, :]) - Q[s_model, a_model])

            s = s_prime

    # Return the learned Q-table and the model
    return Q, model, history
