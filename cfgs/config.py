#!/usr/bin/env python
# -*- coding: utf-8 -*-

cfg = dict(
    # Seed
    np_seed = 0,
    torch_seed = 0,

    # Env Param
    stage_name = 'SuperMarioBros-2-2-v0',
    skip_numb = 4,
    stack_frame_numb = 8,
    resize_shape = 84,

    # Model Param
    check_freq_numb = 10000,
    total_timestep_numb = 1000000,
    # total_timestep_numb = 3000000,
    learning_rate = 0.0001,
    gae = 1.0,
    ent_coef = 0.01,
    n_setps = 512,
    gamma = 0.99,
    batch_size = 64,
    n_episodes = 10000,
    exploration_rate_decay = 0.99999975,

    # Test Param
    episode_numbers = 20,
    max_timestep_test = 1000,
)