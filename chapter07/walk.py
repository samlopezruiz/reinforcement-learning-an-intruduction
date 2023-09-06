from __future__ import annotations
import gymnasium as gym
from typing import Any

from gymnasium.utils.seeding import np_random


class WalkRightEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    reward_range = (-1, -1)

    def __init__(self, size=5):
        self.observation_space = gym.spaces.Discrete(size)
        self.np_random = None
        self.position = 0
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)

        ## action==0: stay in the same state
        ## action==1: move to the right
        self.position += action
        assert self.observation_space.contains(self.position)

        if self.position == self.observation_space.n - 1:
            return self.position, 1, True, {}
        else:
            return self.position, 0, False, {}

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        self.position = 0
        return self.position

    def render(self, mode='ansi'):
        letters = [chr(ord('A') + i) for i in range(self.observation_space.n)]
        letters[self.position] = "#"
        string = "-".join(letters)
        print(string)

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]


gym.envs.registration.register(
    id='WalkRightEnv-v0',
    entry_point=lambda size: WalkRightEnv(size),
    nondeterministic=True,
    kwargs={'size': 5}
)
