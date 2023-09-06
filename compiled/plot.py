from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import plotly.express as px

pio.renderers.default = "browser"

def plot_2d_animated_heatmap(hist):
    """
    :param hist: list of matrices
    :return:
    """

    frames = []
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": dict(duration=5, easing='linear'),
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [
                [k],
                {"frame": dict(duration=500, redraw=True),
                 "mode": "immediate",
                 "transition": dict(duration=0, easing='linear')}
            ],
                "label": k,
                "method": "animate"} for k in range(len(hist))
        ]
    }

    for i, matrix in enumerate(hist):
        frames.append(go.Frame(data=[go.Heatmap(z=matrix, colorscale='viridis'),
                                     # go.Scatter(x=[7], y=[3],
                                     #            mode='markers',
                                     #            marker=dict(size=10, color='black'))
                                     ],
                               name=str(i),
                               # traces=[3],
                               layout=go.Layout(title_text=f'Iteration {i}')))

    matrix = hist[0]
    fig = go.Figure(
        data=[go.Heatmap(z=matrix, colorscale='viridis'),
              # go.Scatter(x=[7], y=[3],
              #            mode='markers',
              #            marker=dict(size=10, color='black'))
              ],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, dict(frame=dict(duration=500, redraw=True),
                                               fromcurrent=True,
                                               transition=dict(duration=0, easing='linear')
                                               )]
                              ),
                         dict(label="Pause",
                              method="animate",
                              args=[[None],
                                    dict(frame=dict(duration=0, redraw=False),
                                         mode='immediate',
                                         transition=dict(duration=0)
                                         )],
                              )])]
        )
    )

    fig.frames = frames

    fig.update_layout(template='plotly_white', sliders=[sliders_dict])
    fig.show()


def plot_3d_animated_mesh(hist):
    """
    :param hist: list of matrices
    :return:
    """

    frames = []
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": dict(duration=5, easing='linear'),
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [
                [k],
                {"frame": dict(duration=500, redraw=True),
                 "mode": "immediate",
                 "transition": dict(duration=0, easing='linear')}
            ],
                "label": k,
                "method": "animate"} for k in range(len(hist))
        ]
    }

    for i, matrix in enumerate(hist):
        x, y = np.array(list(np.ndindex(*matrix.shape))).T
        z = matrix.flatten()
        frames.append(go.Frame(data=[go.Mesh3d(x=x, y=y, z=z, opacity=.9,
                                               colorscale='viridis', intensity=z)],
                               name=str(i),
                               # traces=[3],
                               layout=go.Layout(title_text=f'Iteration {i}')))

    matrix = hist[0]
    x, y = np.array(list(np.ndindex(*matrix.shape))).T
    z = matrix.flatten()
    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, opacity=.9,
                        colorscale='viridis', intensity=z)],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, dict(frame=dict(duration=500, redraw=True),
                                               fromcurrent=True,
                                               transition=dict(duration=0, easing='linear')
                                               )]
                              ),
                         dict(label="Pause",
                              method="animate",
                              args=[[None],
                                    dict(frame=dict(duration=0, redraw=False),
                                         mode='immediate',
                                         transition=dict(duration=0)
                                         )],
                              )])]
        )
    )

    fig.frames = frames

    fig.update_layout(template='plotly_white', sliders=[sliders_dict])
    fig.show()


def plot_windy_results(env, q, policy):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Optimal Value Function and Policy")

    q = np.copy(q)
    unvisited = np.where(q == 0)
    q[unvisited] = -np.inf
    v = np.max(q, axis=1).reshape(env.observation_space.nvec)
    ax.imshow(v.T, origin='lower')

    a_stars = np.argmax(policy, axis=1)
    arrows = np.array([env.actions[a] for a in a_stars])
    arrows[unvisited[0], :] = 0
    arrows = arrows.reshape([*env.observation_space.nvec, 2])
    xr = np.arange(env.observation_space.nvec[0])
    yr = np.arange(env.observation_space.nvec[1])
    ax.quiver(xr, yr, arrows[:, :, 0].T, arrows[:, :, 1].T, pivot='mid')