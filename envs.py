#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gymは、Open AIのRL用ツールキットです
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# OpenAI Gym用に使うNES エミュレーター
from nes_py.wrappers import JoypadSpace

#OpenAI Gymのスーパー・マリオ・ブラザーズの環境
import gym_super_mario_bros

class MarioEnv():
    # スーパー・マリオの環境を初期化
    env = gym_super_mario_bros.make("SuperMarioBros-2-2-v0")

    # 行動空間を以下に制限
    #   0. 右に歩く
    #   1. 右方向にジャンプ
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")