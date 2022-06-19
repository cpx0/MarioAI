#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
import random, datetime, os, argparse

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
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from cfgs import cfg_factory
from envs import SkipFrame, GrayScaleObservation, ResizeObservation
from libs.agents import agent_factory
from libs.logger import MetricLogger


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=1,)
    # parse.add_argument('--port', dest='port', type=int, default=2980,)
    parse.add_argument('--agent', dest='agent', type=str, default='swim',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()


def set_random_seed(torch_seed, np_seed, rand_seed):
    ## fix all random seeds
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')


def train(cfg):
    # スーパー・マリオの環境を初期化
    env = gym_super_mario_bros.make(cfg.stage_name)

    # 行動空間を以下に制限
    #   0. 右に歩く
    #   1. 右方向にジャンプ
    # env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = JoypadSpace(env, cfg.movement)

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    # 環境にWrapperを適用
    env = SkipFrame(env, skip=cfg.skip_numb)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=cfg.resize_shape)
    env = FrameStack(env, num_stack=cfg.stack_frame_numb)

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = agent_factory[cfg.agent_type](state_dim=(cfg.stack_frame_numb, cfg.resize_shape, cfg.resize_shape), action_dim=env.action_space.n, save_dir=save_dir, cfg=cfg)

    logger = MetricLogger(save_dir, cfg)

    episodes = cfg.n_episodes
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


def main():
    args = parse_args()
    cfg = cfg_factory[args.agent]
    set_random_seed(cfg.torch_seed, cfg.np_seed, cfg.rand_seed)
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    # torch.cuda.set_device(args.local_rank) if use_cuda else None
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    train(cfg)


if __name__ == "__main__":
    main()
