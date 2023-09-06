from __future__ import division, print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import plotly.express as px

pio.renderers.default = "browser"


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)






def plot_rewards(totalrewards, rolling_period=None):
    r = pd.Series(totalrewards)

    y = r.rolling(100, min_periods=1).mean() if rolling_period is not None else r
    t = f'rolling {rolling_period} mean reward' if rolling_period is not None else 'rewards'
    fig = px.line(y=y)
    fig.update_layout(xaxis_title='episodes', yaxis_title=t)
    fig.show()


def plot_loss_hist(losses):
    fig = go.Figure()

    for i, loss in enumerate(losses):
        fig.add_trace(go.Scatter(
            x=np.arange(len(loss)),
            y=loss,
            mode='lines',
            name=f'Episode {i + 1}'
        ))

    fig.show()

def plot_cost_to_go(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)

  fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

  fig.update_layout(title='Cost-to-go', autosize=False,
                    width=500, height=500,
                    scene=dict(
                        xaxis_title='Position',
                        yaxis_title='Velocity',
                        zaxis_title='-V(s)'
                    ),
                    margin=dict(l=65, r=50, b=65, t=90))

  fig.show()
  # fig = plt.figure(figsize=(10, 5))
  # ax = fig.add_subplot(111, projection='3d')
  # surf = ax.plot_surface(X, Y, Z,
  #   rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  # ax.set_xlabel('Position')
  # ax.set_ylabel('Velocity')
  # ax.set_zlabel('Cost-To-Go == -V(s)')
  # ax.set_title("Cost-To-Go Function")
  # fig.colorbar(surf)
  # plt.show()