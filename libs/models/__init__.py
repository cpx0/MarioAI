#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .simple import MarioNet


model_factory = {
    'mario': MarioNet,
}