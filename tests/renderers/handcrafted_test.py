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
"""Tests for handcrafted."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from six.moves import range

from spriteworld import constants as const
from spriteworld import sprite as sprite_lib
from spriteworld.renderers import handcrafted


class SpriteFactorsTest(parameterized.TestCase):

  def testWrongFactors(self):
    handcrafted.SpriteFactors(factors=('x', 'y', 'scale'))
    with self.assertRaises(ValueError):
      handcrafted.SpriteFactors(factors=('position', 'scale'))
    with self.assertRaises(ValueError):
      handcrafted.SpriteFactors(factors=('x', 'y', 'size'))

  def testSingleton(self):
    sprite = sprite_lib.Sprite(
        x=0.1, y=0.3, shape='square', scale=0.5, c0=0, c1=0, c2=255)

    renderer = handcrafted.SpriteFactors()
    renderer.render(sprites=[sprite])

  def testSequence(self):
    sprites = [
        sprite_lib.Sprite(x=np.random.rand(), y=np.random.rand())
        for _ in range(5)
    ]
    renderer = handcrafted.SpriteFactors()
    renderer.render(sprites=sprites)

  @parameterized.parameters(1, 2, 5)
  def testOutputLength(self, num_sprites):
    sprites = [sprite_lib.Sprite() for _ in range(num_sprites)]
    renderer = handcrafted.SpriteFactors()
    outputs = renderer.render(sprites=sprites)
    self.assertLen(outputs, num_sprites)

  @parameterized.parameters((1, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape',
                                 'angle', 'x_vel', 'y_vel')),
                            (1, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape')),
                            (2, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape')),
                            (5, ('x', 'y', 'angle', 'x_vel', 'y_vel')))
  def testFactorSubset(self, num_sprites, factors):
    sprites = [sprite_lib.Sprite() for _ in range(num_sprites)]
    renderer = handcrafted.SpriteFactors(factors=factors)
    outputs = renderer.render(sprites=sprites)
    output_keys = [set(x) for x in outputs]
    self.assertSequenceEqual(output_keys, num_sprites * [set(factors)])

  @parameterized.parameters((1, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape',
                                 'angle', 'x_vel', 'y_vel')),
                            (1, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape')),
                            (2, ('x', 'y', 'scale', 'c0', 'c1', 'c2', 'shape')),
                            (5, ('x', 'y', 'angle', 'x_vel', 'y_vel')))
  def testObservationSpec(self, num_sprites, factors):
    sprites = [sprite_lib.Sprite() for _ in range(num_sprites)]
    renderer = handcrafted.SpriteFactors(factors=factors)
    renderer.render(sprites=sprites)
    obs_spec = renderer.observation_spec()

    for v in obs_spec[0].values():
      self.assertEqual(v.shape, ())

    obs_spec_keys = [set(x) for x in obs_spec]
    self.assertSequenceEqual(obs_spec_keys, num_sprites * [set(factors)])

  @parameterized.parameters(
      (0.5, 0.5, 'square', 0, 0, 255, 0.5, 0),
      (0.5, 0.5, 'square', 255, 0, 0, 0.5, 0),
      (0.5, 0.8, 'octagon', 0.4, 0.8, 0.5, 0.6, 90),
      (0.5, 0.3, 'star_5', 180, 180, 0, 0.2, 240),
  )
  def testAttributesSingleton(self, x, y, shape, c0, c1, c2, scale, angle):
    sprite = sprite_lib.Sprite(
        x=x, y=y, shape=shape, c0=c0, c1=c1, c2=c2, scale=scale, angle=angle)
    renderer = handcrafted.SpriteFactors()
    outputs = renderer.render(sprites=[sprite])[0]

    self.assertEqual(outputs['shape'], const.ShapeType[shape].value)
    for (name, value) in (('x', x), ('y', y), ('c0', c0), ('c1', c1),
                          ('c2', c2), ('scale', scale), ('angle', angle)):
      self.assertAlmostEqual(outputs[name], value, delta=1e-4)

  def testAttributesTwoSprites(self):
    x = [0.5, 0.3]
    y = [0.4, 0.8]
    shape = ['square', 'spoke_4']
    c0 = [0, 200]
    c1 = [255, 100]
    c2 = [0, 200]
    scale = [0.2, 0.3]
    angle = [0, 120]
    x_vel = [0.0, 0.1]
    y_vel = [-0.2, 0.05]

    sprites = []
    for i in range(2):
      sprites.append(
          sprite_lib.Sprite(
              x=x[i],
              y=y[i],
              shape=shape[i],
              c0=c0[i],
              c1=c1[i],
              c2=c2[i],
              scale=scale[i],
              angle=angle[i],
              x_vel=x_vel[i],
              y_vel=y_vel[i]))

    renderer = handcrafted.SpriteFactors()
    outputs = renderer.render(sprites=sprites)

    for i in range(2):
      self.assertEqual(outputs[i]['shape'], const.ShapeType[shape[i]].value)
      for (name, value) in (('x', x), ('y', y), ('c0', c0), ('c1', c1),
                            ('c2', c2), ('scale', scale), ('angle', angle),
                            ('x_vel', x_vel), ('y_vel', y_vel)):
        self.assertAlmostEqual(outputs[i][name], value[i], delta=1e-4)


class SpritePassthroughTest(parameterized.TestCase):

  def testRenderOne(self):
    sprite = sprite_lib.Sprite(
        x=0.1, y=0.3, shape='square', scale=0.5, c0=0, c1=0, c2=255)

    renderer = handcrafted.SpritePassthrough()
    observation = renderer.render(sprites=[sprite])
    self.assertEqual(observation, [sprite])

  @parameterized.parameters((3,), (5,), (10,))
  def testKSprites(self, num_sprites):
    sprites = [
        sprite_lib.Sprite(x=np.random.rand(), y=np.random.rand())
        for _ in range(num_sprites)
    ]

    renderer = handcrafted.SpritePassthrough()
    observation = renderer.render(sprites=sprites)
    self.assertSequenceEqual(observation, sprites)

    obs_spec = renderer.observation_spec()
    self.assertTrue(obs_spec.shape, (num_sprites,))


class SuccessTest(absltest.TestCase):

  def testRender(self):
    renderer = handcrafted.Success()
    self.assertTrue(renderer.render(global_state={'success': True}))
    self.assertFalse(renderer.render(global_state={'success': False}))
    with self.assertRaises(KeyError):
      renderer.render(global_state={})


if __name__ == '__main__':
  absltest.main()
