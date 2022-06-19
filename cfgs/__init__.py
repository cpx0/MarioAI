#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .swim import cfg as swim_cfg


class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    swim = cfg_dict(swim_cfg),
)