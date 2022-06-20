#!/usr/bin/env python
# -*- coding: utf-8 -*-

cfg = dict(
    # Env Param
    stage_name = 'SuperMarioBros-2-2-v0',
    skip_numb = 4,
    stack_frame_numb = 4,
    resize_shape = 84,

    # Agent Param
    agent_type = 'swim',
    movement = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ],
    deque_size = 20000,         # 100000: CUDAのcacheサイズ

    # Model Param
    model_type = 'simple',
    check_freq_numb = 10000,
    total_timestep_numb = 1000000,
    # total_timestep_numb = 3000000,
    learning_rate = 0.0001,     # 0.00025
    gae = 1.0,
    ent_coef = 0.01,
    n_setps = 512,
    gamma = 0.99,               # 0.9
    batch_size = 32,            # 64
    n_episodes = 300000,        # First 100 episodes take 5 minutes.
    # exploration_rate_init * exploration_rate_decay**n = exploration_rate_min
    # ε-greedy法によってε(exploration_rate)が最小値まで減衰するステップ数 n について上式を解くと、
    # n=(log0.1)/(log(exploration_rate_decay))≒9,210,339.220683588
    # よって、9,210,340回目のステップ以降は、exploration_rate=0.1で固定となる
    exploration_rate_init = 1,
    exploration_rate_decay = 0.99999975,
    exploration_rate_min = 0.1,
    burnin_n_step = 1e4,        # 1e4: 経験を訓練させるために最低限必要なステップ数
    learn_every_n_step = 3,     # 3  : Q_onlineを更新するタイミングを示すステップ数
    sync_every_n_step = 1e4,    # 1e4: Q_target & Q_onlineを同期させるタイミングを示すステップ数
    save_every_n_step = 5e5,    # 5e5: Netを保存するまでの実験ステップの数です

    # Seed
    random_seed = 0,
    np_seed = 0,
    torch_seed = 0,

    # Test Param
    episode_numbers = 20,
    max_timestep_test = 1000,
)
