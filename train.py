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

# Import the SIMPLIFIED controls
from gym.spaces import Box
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from envs import SkipFrame, GrayScaleObservation, ResizeObservation
from cfgs import cfg_factory
from libs.agents import agent_factory
from libs.logger import MetricLogger


def parse_args():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--port', dest='port', type=int, default=2980,)
    parse.add_argument('--agent', dest='agent', type=str, default='mario',)
    parse.add_argument('--model', dest='model', type=str, default='mario',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.agent]


def train():
    # スーパー・マリオの環境を初期化
    env = gym_super_mario_bros.make(cfg.stage_name)

    # 行動空間を以下に制限
    #   0. 右に歩く
    #   1. 右方向にジャンプ
    # env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    # 環境にWrapperを適用
    env = SkipFrame(env, skip=cfg.skip_numb)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=cfg.resize_shape)
    env = FrameStack(env, num_stack=cfg.stack_frame_numb)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}\n")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, cfg.resize_shape, cfg.resize_shape), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = cfg.n_episodes # 元々は10でしたが、日本語版では少し伸ばしてみましょう。5分程度かかります
    for e in range(episodes):

        state = env.reset()

        # ゲーム開始！
        while True:
            # 現在の状態に対するエージェントの行動を決める
            action = mario.act(state)
            # エージェントが行動を実行
            next_state, reward, done, info = env.step(action)
            # 記憶
            mario.cache(state, next_state, action, reward, done)
            # 訓練
            q, loss = mario.learn()
            # ログ保存
            logger.log_step(reward, loss, q)
            # 状態の更新
            state = next_state
            # ゲームが終了したかどうかを確認
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)