#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .config import cfg as mario_cfg


class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    mario=cfg_dict(mario_cfg),
)