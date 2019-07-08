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
"""Tests for shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
from PIL import ImageDraw
from spriteworld import shapes


class ShapesTest(parameterized.TestCase):

  def _test_area(self, path):
    im_size = 1000
    path = im_size * path / 2 + im_size / 2
    im = Image.new('RGB', (im_size, im_size))
    draw = ImageDraw.Draw(im)
    draw.polygon([tuple(p) for p in path], fill=(255, 255, 255))
    desired_area = 0.25 * im_size * im_size * 3
    true_area = np.sum(np.array(im) > 0)
    self.assertAlmostEqual(desired_area / true_area, 1, delta=1e-2)

  @parameterized.parameters(3, 4, 5, 6, 7, 8, 10)
  def testPolygon(self, num_sides):
    path = shapes.polygon(num_sides)
    self._test_area(path)

  @parameterized.parameters((3, 0.5), (3, 1.5), (5, 0.6), (5, 2.0), (8, 0.2),
                            (8, 3.0), (11, 1.2))
  def testStar(self, num_sides, point_height):
    path = shapes.star(num_sides, point_height)
    self._test_area(path)

  @parameterized.parameters((3, 0.5), (3, 1.5), (5, 0.6), (5, 2.0), (8, 0.2),
                            (8, 3.0), (11, 1.2))
  def testSpokes(self, num_sides, spoke_height):
    path = shapes.star(num_sides, spoke_height)
    self._test_area(path)


if __name__ == '__main__':
  absltest.main()
