import gym
from kaggle_environments import evaluate, make
from random import choice
import random
import numpy as np


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5):
        super(ConnectX, self).__init__()
        self.env = make('connectx', debug=True)
        self.pair = [None, 'negamax']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        N_CHANNELS = 1
        HEIGHT = 1
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(N_CHANNELS, HEIGHT, config.columns * config.rows), dtype=np.uint8)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        if type(action) is np.int64:
            action = int(action)
        res = self.trainer.step(action)

        obs = np.array(res[0]['board'])
        obs = np.expand_dims(obs, axis=0)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.uint8)

        reward = res[1]
        done = res[2]
        info = res[3]

        return obs, reward, done, info

    def reset(self):
        print("reset----------------------------------------------------")

        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        res = self.trainer.reset()

        obs = np.array(res['board'])
        obs = np.expand_dims(obs, axis=0)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.uint8)
        return obs

    def render(self, **kwargs):
        return self.env.render(**kwargs)