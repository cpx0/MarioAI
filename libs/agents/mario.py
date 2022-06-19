#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from collections import deque
import random, os

from libs.models import model_factory


class Mario:
    def __init__():
        pass

    def act(self, state):
        """状態が与えられたとき、ε-greedy法に従って行動を選択します"""
        pass

    def cache(self, experience):
        """経験をメモリに追加します"""
        pass

    def recall(self):
        """記憶からの経験のサンプリングします"""
        pass

    def learn(self):
        """経験のデータのバッチで、オンラインに行動価値関数(Q)を更新します"""
        pass


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.cfg = cfg

        self.use_cuda = torch.cuda.is_available()

        # 最適な行動を予測するマリオ用のDNNです。「訓練」セクションで実装します
        self.net = model_factory[self.cfg.model_type](self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  #  Netを保存するまでの実験ステップの数です

    def act(self, state):
        """
        状態が与えられると、ε-greedy法で行動を選択し、ステップの値を更新します

        Inputs:
            state(LazyFrame):現在の状態における一つの観測オブジェクトで、(state_dim)次元となります
        Outputs:
            action_idx (int): マリオが取る行動を示す整数値です
        """
        # 探索（EXPLORE）
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 活用（EXPLOIT）
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # exploration_rateを減衰させます
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # ステップを+1します
        self.curr_step += 1
        return action_idx


class Mario(Mario):  # さきほどのクラスのサブクラスとなっています
    def __init__(self, state_dim, action_dim, save_dir, cfg):
        super().__init__(state_dim, action_dim, save_dir, cfg)
        self.memory = deque(maxlen=cfg.deque_size)
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = cfg.burnin  # 経験を訓練させるために最低限必要なステップ数
        self.learn_every = cfg.learn_every  # Q_onlineを更新するタイミングを示すステップ数
        self.sync_every = cfg.sync_every  # Q_target & Q_onlineを同期させるタイミングを示すステップ数

    def cache(self, state, next_state, action, reward, done):
        """
        経験をself.memory (replay buffer)に保存します

        Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        メモリから経験のバッチを取得します
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, chkpt_path):
        self.net.load_state_dict(torch.load(chkpt_path)["model"])
        self.exploration_rate = torch.load(chkpt_path)["exploration_rate"]
        print(f"MarioNet loaded from {chkpt_path}")
    
    def load_target(self, chkpt_path):
        self.net.load_state_dict(torch.load(chkpt_path)["model"])
        self.exploration_rate = 0.0
        print(f"MarioNet loaded from {chkpt_path}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        # メモリからサンプリング
        state, next_state, action, reward, done = self.recall()
        # TD Estimateの取得
        td_est = self.td_estimate(state, action)
        # TD Targetの取得
        td_tgt = self.td_target(reward, next_state, done)
        # 損失をQ_onlineに逆伝播させる
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

