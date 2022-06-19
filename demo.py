#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 日本語版追加：訓練後の様子
from pathlib import Path

from cfgs import cfg_factory
from libs.utils import *
from train import parse_args


def main():
    args = parse_args()
    cfg = cfg_factory[args.agent]
    # TODO: User should be refactored instead of hard coded following dir and name
    chkpt_dir = Path("checkpoints") / "2022-06-18T16-35-14"
    chkpt_name = args.finetune_from if args.finetune_from else "mario_net_9.chkpt"
    img, img_color = display_color_image(cfg, chkpt_dir, chkpt_name)
    display_grayscale_movie(img, cfg, chkpt_dir, chkpt_name)
    display_color_movie(img_color, cfg, chkpt_dir, chkpt_name)

if __name__ == "__main__":
    main()
