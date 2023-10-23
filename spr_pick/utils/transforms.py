import spr_pick
from typing import NewType


Transform = NewType("Transform", object)
"""Typing label for otherwise undefined Transform type.
"""


class NoiseTransform(object):
    def __init__(self, style: str):
        self.style = style

    def __call__(self, imgs):
        imgs, _ = spr_pick.utils.noise.add_style(imgs, self.style)
        return imgs
