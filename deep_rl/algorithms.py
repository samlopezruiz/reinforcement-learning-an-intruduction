import numpy as np
from tqdm import tqdm

from utils.plot import save_frames_as_gif


def play_episode(env,
                 policy,
                 return_hist=False,
                 save_animation=False,
                 path='./',
                 filename='gym_animation.gif',
                 max_iter=2000,
                 verbose=True):
  obs, rs, acts = [], [], []
  observation, _ = env.reset()
  done = False
  frames = []
  iters = 0

  pbar = tqdm(total=max_iter, desc='Running episode', disable=not verbose, position=0)
  while not done and iters < max_iter:
    # Render to frames buffer
    if save_animation:
      frames.append(env.render())
    action = policy(observation)

    if return_hist:
      obs.append(observation)
      acts.append(action)

    observation, reward, done, reset, info = env.step(action)

    if return_hist:
      rs.append(reward)

    iters += 1
    pbar.update(1)



  if save_animation:
    save_frames_as_gif(frames, path, filename)

  if return_hist:
    return {'observations': np.array(obs), 'rewards': np.array(rs), 'actions': np.array(acts)}

def play_one(env, model, eps, gamma):
  observation, _ = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, _, info = env.step(action)

    if done:
      reward = -200

    # update the model
    next = model.predict(observation)
    # print(next.shape)
    assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward

def play_pg_one_td(env, pmodel, vmodel, gamma, max_iter=2000, verbose=False, pbar_prefix=None):
  observation, _ = env.reset()
  done = False
  rewards = []
  iters = 0
  losses = []

  pbar_prefix = 'Policy Gradient' if pbar_prefix is None else pbar_prefix
  pbar = tqdm(total=max_iter, desc=pbar_prefix, disable=not verbose, position=0)

  while not done and iters < max_iter:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, _, info = env.step(action)

    rewards.append(reward)

    # update the models
    V_next = vmodel.predict(observation)
    G = reward + gamma * V_next
    advantage = G - vmodel.predict(prev_observation)
    loss = pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)

    losses.append(loss.numpy())

    iters += 1
    pbar.update(1)
    pbar.set_description('{}, Mean reward={:.1f}'.format(pbar_prefix, np.mean(rewards)))

  return rewards, np.array(losses)

def play_one_render(env, pmodel):
  obs, _ = env.reset()
  done = False

  while not done:

    env.render()

    # Predict action using Policy Model
    action = pmodel.sample_action(obs)

    # Execute the action in the environment
    obs, reward, done, _, info = env.step(action)

  env.close()