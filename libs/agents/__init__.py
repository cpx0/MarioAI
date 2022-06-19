#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .mario import Mario as Swimmer


agent_factory = {
    'swim': Swimmer,
}