import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_learning(histories, labels):
    with matplotlib.rc_context({'figure.figsize': [7, 7]}):
        # plt.plot([0, 1])
        plt.figure()
        plt.title(f"Cumulative Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Cumulative Reward")

        plt.xlim([0, 6_000])
        plt.ylim([0, 400])

        for history, label in zip(histories, labels):
            plt.plot(np.cumsum(history), label=label)

        plt.legend()
        plt.show()


def plot_results(env, q, label):
    # Render example episode
    done, (state, _) = False, env.reset()
    env.render()
    while not done:
        # action = np.random.choice(env.action_space.n, p=q[state] / np.sum(q[state]))
        action = np.argmax(q[state])
        state, reward, done, _, info = env.step(action)
        env.render()
    plt.title(f"Example Episode ({label})")

    env = env.unwrapped
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(f"Value Function and Policy ({label})")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot value function
    q = np.copy(q)
    unvisited = np.where(q == 0)
    q[unvisited] = -np.inf
    v = np.max(q, axis=1).reshape(env.observation_space.nvec)
    ax.imshow(v.T, origin='lower')

    # Plot actions of the policy
    a_stars = np.argmax(q, axis=1)
    arrows = np.array([env.actions[a] for a in a_stars])
    arrows[unvisited[0], :] = 0
    arrows = arrows.reshape([*env.observation_space.nvec, 2])
    xr = np.arange(env.observation_space.nvec[0])
    yr = np.arange(env.observation_space.nvec[1])
    ax.quiver(xr, yr, arrows[:, :, 0].T, arrows[:, :, 1].T, pivot='mid', scale=20)
