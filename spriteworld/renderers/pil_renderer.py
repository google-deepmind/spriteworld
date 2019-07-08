# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Python Image Library (PIL/Pillow) renderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np
from PIL import Image
from PIL import ImageDraw
from spriteworld.renderers import abstract_renderer


class PILRenderer(abstract_renderer.AbstractRenderer):
  """Render using Python Image Library (PIL/Pillow)."""

  def __init__(self,
               image_size=(64, 64),
               anti_aliasing=1,
               bg_color=None,
               color_to_rgb=None):
    """Construct PIL renderer.

    Args:
      image_size: Int tuple (height, width). Size of output of .render().
      anti_aliasing: Int. Anti-aliasing factor. Linearly scales the size of the
        internal canvas.
      bg_color: None or 3-tuple of ints in [0, 255]. Background color. If None,
        background is (0, 0, 0).
      color_to_rgb: Callable converting a tuple (c1, c2, c3) to a uint8 tuple
        (r, g, b) in [0, 255].
    """
    self._image_size = image_size
    self._anti_aliasing = anti_aliasing
    self._canvas_size = (anti_aliasing * image_size[0],
                         anti_aliasing * image_size[1])

    if color_to_rgb is None:
      color_to_rgb = lambda x: x
    self._color_to_rgb = color_to_rgb

    if bg_color is None:
      bg_color = (0, 0, 0)
    self._canvas_bg = Image.new('RGB', self._canvas_size, bg_color)

    self._observation_spec = specs.Array(
        shape=self._image_size + (3,), dtype=np.uint8)

    self._canvas = Image.new('RGB', self._canvas_size)
    self._draw = ImageDraw.Draw(self._canvas)

  def render(self, sprites=(), global_state=None):
    """Render sprites.

    Sprites are ordered from background to foreground.

    Args:
      sprites: Iterable of sprite.Sprite instances.
      global_state: Unused global state.

    Returns:
      Numpy uint8 RGB array of size self._image_size + (3,).
    """
    self._canvas.paste(self._canvas_bg)
    for obj in sprites:
      vertices = self._canvas_size * obj.vertices
      color = self._color_to_rgb(obj.color)
      self._draw.polygon([tuple(v) for v in vertices], fill=color)
    image = self._canvas.resize(self._image_size, resample=Image.ANTIALIAS)
    return np.array(image)

  def observation_spec(self):
    return self._observation_spec
