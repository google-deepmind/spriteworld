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
"""Tests for sprite_generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from spriteworld import factor_distributions as distribs
from spriteworld import sprite
from spriteworld import sprite_generators

_distrib_0 = distribs.Product([
    distribs.Discrete('x', [0.5]),
    distribs.Discrete('y', [0.5]),
    distribs.Discrete('shape', ['square', 'triangle']),
    distribs.Discrete('c0', [255]),
    distribs.Discrete('c1', [255]),
    distribs.Discrete('c2', [255]),
])

_distrib_1 = distribs.Product([
    distribs.Discrete('x', [0.5]),
    distribs.Discrete('y', [0.5]),
    distribs.Discrete('shape', ['hexagon', 'circle', 'star_5']),
    distribs.Discrete('c0', [255]),
    distribs.Discrete('c1', [255]),
    distribs.Discrete('c2', [255]),
])


class SpriteGeneratorTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 5)
  def testGenerateSpritesLengthType(self, num_sprites):
    g = sprite_generators.generate_sprites(_distrib_0, num_sprites=num_sprites)
    sprite_list = g()
    self.assertIsInstance(sprite_list, list)
    self.assertLen(sprite_list, num_sprites)
    self.assertIsInstance(sprite_list[0], sprite.Sprite)

  def testGenerateSpritesCallableNum(self):
    minval = 3
    maxval = 6
    num_sprites = np.random.randint(minval, maxval)
    g = sprite_generators.generate_sprites(_distrib_0, num_sprites=num_sprites)
    sprite_list = g()
    self.assertGreaterEqual(len(sprite_list), minval)
    self.assertLess(len(sprite_list), maxval)


class ChainGeneratorsTest(absltest.TestCase):

  def testOutput(self):
    g_0 = sprite_generators.generate_sprites(_distrib_0, num_sprites=1)
    g_1 = sprite_generators.generate_sprites(_distrib_1, num_sprites=2)
    g_chain = sprite_generators.chain_generators(g_0, g_1)
    sprite_list = g_chain()
    self.assertIsInstance(sprite_list, list)
    self.assertLen(sprite_list, 3)
    self.assertTrue(_distrib_0.contains(sprite_list[0].factors))
    self.assertTrue(_distrib_1.contains(sprite_list[1].factors))
    self.assertTrue(_distrib_1.contains(sprite_list[2].factors))


class SampleGeneratorTest(absltest.TestCase):

  def testOutput(self):
    g_0 = sprite_generators.generate_sprites(_distrib_0, num_sprites=1)
    g_1 = sprite_generators.generate_sprites(_distrib_1, num_sprites=1)
    g_chain = sprite_generators.sample_generator((g_0, g_1))
    sprite_list = g_chain()
    self.assertIsInstance(sprite_list, list)
    self.assertLen(sprite_list, 1)
    self.assertNotEqual(
        _distrib_0.contains(sprite_list[0].factors),
        _distrib_1.contains(sprite_list[0].factors))


class ShuffleTest(absltest.TestCase):

  def testOutput(self):
    g = sprite_generators.generate_sprites(_distrib_0, num_sprites=5)
    g_shuffle = sprite_generators.shuffle(g)
    sprite_list = g_shuffle()
    self.assertIsInstance(sprite_list, list)
    self.assertLen(sprite_list, 5)


if __name__ == '__main__':
  absltest.main()
