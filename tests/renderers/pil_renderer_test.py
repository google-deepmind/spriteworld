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
"""Tests for pil_renderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys
from absl.testing import absltest
import numpy as np
from spriteworld import sprite
from spriteworld.renderers import pil_renderer


class PilRendererTest(absltest.TestCase):

  def _get_sprites(self):
    """Get list of sprites."""
    sprites = [
        sprite.Sprite(
            x=0.75, y=0.95, shape='spoke_6', scale=0.2, c0=20, c1=50, c2=80),
        sprite.Sprite(
            x=0.2, y=0.3, shape='triangle', scale=0.1, c0=150, c1=255, c2=100),
        sprite.Sprite(
            x=0.7, y=0.5, shape='square', scale=0.3, c0=0, c1=255, c2=0),
        sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=0.3, c0=255, c1=0, c2=0),
    ]
    return sprites

  def testBasicFunctionality(self):
    renderer = pil_renderer.PILRenderer(image_size=(64, 64))
    renderer.render(self._get_sprites())

  def testBackground(self):
    bg_color = (5, 6, 7)
    renderer = pil_renderer.PILRenderer(image_size=(64, 64), bg_color=bg_color)
    image = renderer.render(self._get_sprites())
    self.assertSequenceEqual(list(image[5, 5]), bg_color)

  def testOcclusion(self):
    renderer = pil_renderer.PILRenderer(image_size=(64, 64))
    image = renderer.render(self._get_sprites())
    self.assertSequenceEqual(list(image[32, 32]), [255, 0, 0])
    self.assertSequenceEqual(list(image[32, 50]), [0, 255, 0])

  def testAntiAliasing(self):
    renderer = pil_renderer.PILRenderer(image_size=(16, 16), anti_aliasing=5)
    image = renderer.render(self._get_sprites())

    self.assertSequenceEqual(list(image[4, 6]), [0, 0, 0])
    self.assertSequenceEqual(list(image[6, 6]), [255, 0, 0])
    # Python2 and Python3 give slightly different anti-aliasing, so we specify
    # bounds for border values:
    self.assertTrue(all(image[5, 6] >= [50, 0, 0]))
    self.assertTrue(all(image[5, 6] <= [120, 30, 0]))
    self.assertTrue(all(image[7, 6] >= [200, 0, 0]))
    self.assertTrue(all(image[7, 6] <= [255, 50, 0]))

    renderer = pil_renderer.PILRenderer(image_size=(16, 16), anti_aliasing=1)
    image = renderer.render(self._get_sprites())
    self.assertSequenceEqual(list(image[4, 6]), [0, 0, 0])
    self.assertSequenceEqual(list(image[6, 6]), [255, 0, 0])
    self.assertSequenceEqual(list(image[7, 6]), [255, 0, 0])

  def testColorToRGB(self):
    s = sprite.Sprite(x=0.5, y=0.5, shape='square', c0=0.2, c1=0.5, c2=0.5)
    def _color_to_rgb(c):
      return tuple((255 * np.array(colorsys.hsv_to_rgb(*c))).astype(np.uint8))

    renderer = pil_renderer.PILRenderer(
        image_size=(64, 64), color_to_rgb=_color_to_rgb)
    image = renderer.render([s])
    self.assertSequenceEqual(list(image[32, 32]), [114, 127, 63])


if __name__ == '__main__':
  absltest.main()
