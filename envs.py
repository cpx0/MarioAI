#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms as T
import numpy as np
import os, copy

# Gymは、Open AIのRL用ツールキットです
import gym
from gym.spaces import Box

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """スキップした後のフレームのみを返す"""
        super().__init__(env)
        self._skip = skip
        self.last_x_pos = 0
        self.last_y_pos = 0
        self.last_score = 0
        self.last_coins = 0
        self.Last_status = 0

    def step(self, action):
        """行動を繰り返し、報酬を合計する"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # 報酬を蓄積し、同じ行動を繰り返す
            obs, reward, done, info = self.env.step(action)
            total_reward += reward + \
                add_reward(action, info, self.last_x_pos, self.last_y_pos,
                            self.last_score, self.last_coins, self.Last_status)
            self.update_last_info(info)
            if done:
                break
        return obs, total_reward, done, info
    
    def update_last_info(self, curr_info):
        self.last_x_pos = curr_info["x_pos"]
        self.last_y_pos = curr_info["y_pos"]
        self.last_score = curr_info["score"]
        self.last_coins = curr_info["coins"]
        self.Last_status = curr_info["status"]


def add_reward(action, curr_info, last_x, last_y, last_score, last_coins, last_status):
    # give a penalty when an agent acting to go right(1,2,3,4,8,9) or left(6) cannot move
    x_stay_penalty = -1 if abs(curr_info["x_pos"] - last_x) == 0 \
        and (action > 0 and action < 5 or action == (6 or 8 or 9)) else 0
    # give a penalty when an agent acting to go up(2,4,5,7,9,11) or down(10) cannot move
    # y_stay_penalty = -0.5 if abs(curr_info["y_pos"] - last_y) == 0 and action == (2 or 4 or 5 or 7 or 9 or 11) \
    #                     or abs(curr_info["y_pos"] - last_y) == 0 and action == 10 else 0
    y_stay_penalty = 0
    # scoreはステージの最初では確実に右に進まないと獲得できないので、scoreへのrewardで右へのバイアスがかかることを期待
    score_reward = 0.5 if curr_info["score"] > last_score else 0
    # score_reward = 0
    # When getting coins, Mario get score, too. 
    # So, when getting coins, an adding reward is (score_reward + coins_reward).
    # coins_reward = 0.25 if curr_info["coins"] > last_coins else 0
    coins_reward = 0
    if last_status == "fireball":
        status_reward = -0.5 if curr_info["status"] == "tall" or curr_info["status"] == "small" else 0
    elif last_status == "tall":
        status_reward = -0.5 if curr_info["status"] == "small" else (0.5 if curr_info["status"] == "fireball" else 0)
    else:
        status_reward = 0.5 if curr_info["status"] == "fireball" or curr_info["status"] == "tall" else 0
    
    return x_stay_penalty + y_stay_penalty + score_reward + coins_reward + status_reward


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # [H, W, C] のarrayを、[C, H, W] のtensorに変換
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

