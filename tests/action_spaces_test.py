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
"""Tests for action_spaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from spriteworld import action_spaces
from spriteworld import sprite


class SelectMoveTest(parameterized.TestCase):

  @parameterized.parameters((np.array([0.5, 0.5, 0.2, 0.75]), 0.),
                            (np.array([0.5, 0.5, 0.2, 0.75]), 0.2),
                            (np.array([0.2, 0.3, 0.9, 0.05]), 0.2),
                            (np.array([0.5, 0.5, 0.2, 0.75]), 0.5))
  def testNoiseScale(self, action, noise_scale):
    action_space = action_spaces.SelectMove(scale=0.1, noise_scale=noise_scale)
    action_space.apply_noise_to_action(action)

  @parameterized.parameters(
      (1, np.array([0.5, 0.5, 0.2, 0.75]), np.array([-0.3, 0.25])),
      (1, np.array([0.2, 0.5, 0.2, 0.75]), np.array([-0.3, 0.25])),
      (0.5, np.array([0.2, 0.5, 0.2, 0.75]), np.array([-0.15, 0.125])))
  def testGetMotion(self, scale, action, true_motion):
    action_space = action_spaces.SelectMove(scale=scale)
    motion = action_space.get_motion(action)
    self.assertTrue(np.allclose(motion, true_motion, atol=1e-4))

  @parameterized.parameters((1, np.array([0.5, 0.5, 0.2, 0.75]), 0., 0.),
                            (1, np.array([0.5, 0.5, 0.2, 0.75]), 1., -0.39),
                            (1, np.array([0.2, 0.3, 0.2, 0.75]), 1., -0.39),
                            (0.5, np.array([0.5, 0.5, 0.2, 0.75]), 1., -0.195))
  def testMotionCost(self, scale, action, motion_cost, true_cost):
    action_space = action_spaces.SelectMove(
        scale=scale, motion_cost=motion_cost)
    cost = action_space.step(action, sprites=[], keep_in_frame=False)
    self.assertAlmostEqual(cost, true_cost, delta=0.01)

  def testMoveSprites(self):
    """Take a series of actions and repeatedly check sprite motions."""
    action_space = action_spaces.SelectMove(scale=0.5)
    sprites = [sprite.Sprite(x=0.55, y=0.5), sprite.Sprite(x=0.5, y=0.5)]

    # Move second (top) sprite
    action_space.step(
        np.array([0.52, 0.52, 0.5, 0.48]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.55, 0.5], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.58, 0.5, 0.9, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move neither sprite
    action_space.step(
        np.array([0.58, 0.5, 0.9, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move second (top) sprite
    action_space.step(
        np.array([0.5, 0.5, 0.2, 0.5]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.78, 0.74, 0.9, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.95, 0.9], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.92, 0.9, 0.9, 0.5]), sprites, keep_in_frame=True)
    self.assertTrue(np.allclose(sprites[0].position, [1., 0.9], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.98, 0.9, 0.7, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [1.1, 1.1], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))


class DragAndDropTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, np.array([0.5, 0.5, 0.2, 0.75]), np.array([-0.3, 0.25])),
      (1, np.array([0.2, 0.5, -0.2, -0.75]), np.array([-0.4, -1.25])),
      (0.5, np.array([0.2, 0.5, 0.6, 0.3]), np.array([0.2, -0.1])),
      (0.5, np.array([0.2, 0.5, -0.6, -0.3]), np.array([-0.4, -0.4])),
      (0.5, np.array([0.0, 0.0, -0.2, -0.4]), np.array([-0.1, -0.2])))
  def testGetMotion(self, scale, action, true_motion):
    action_space = action_spaces.DragAndDrop(scale=scale)
    motion = action_space.get_motion(action)
    self.assertTrue(np.allclose(motion, true_motion, atol=1e-4))

  def testMoveSprites(self):
    """Take a series of actions and repeatedly check sprite motions."""
    action_space = action_spaces.DragAndDrop(scale=0.5)
    sprites = [sprite.Sprite(x=0.55, y=0.5), sprite.Sprite(x=0.5, y=0.5)]

    # Move second (top) sprite
    action_space.step(
        np.array([0.52, 0.52, 0.52, 0.5]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.55, 0.5], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.58, 0.5, 0.98, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move neither sprite
    action_space.step(
        np.array([0.58, 0.5, 0.9, 0.9]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.5, 0.49], atol=1e-5))

    # Move second (top) sprite
    action_space.step(
        np.array([0.5, 0.5, 0.2, 0.5]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.75, 0.7], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.78, 0.74, 0.98, 0.94]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [0.85, 0.8], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.82, 0.8, 1.3, 1.0]), sprites, keep_in_frame=True)
    self.assertTrue(np.allclose(sprites[0].position, [1., 0.9], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))

    # Move first (bottom) sprite
    action_space.step(
        np.array([0.99, 0.9, 1.19, 1.3]), sprites, keep_in_frame=False)
    self.assertTrue(np.allclose(sprites[0].position, [1.1, 1.1], atol=1e-5))
    self.assertTrue(np.allclose(sprites[1].position, [0.35, 0.49], atol=1e-5))


class EmbodiedTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.1, 0, np.array([0., -0.1])), (0.1, 1, np.array([-0.1, 0.])),
      (0.1, 2, np.array([0., 0.1])), (0.1, 3, np.array([0.1, 0.])),
      (0.5, 3, np.array([0.5, 0.])))
  def testGetMotion(self, step_size, motion_action, true_motion):
    action_space = action_spaces.Embodied(step_size=step_size)
    motion = action_space.action_to_motion[motion_action]
    self.assertTrue(np.allclose(motion, true_motion, atol=1e-5))

  @parameterized.parameters(
      ([[0.5, 0.5], [0.2, 0.8]], (0, 2), [[0.5, 0.5], [0.2, 0.9]]),
      ([[0.5, 0.5], [0.2, 0.8]], (1, 2), [[0.5, 0.5], [0.2, 0.9]]),
      ([[0.5, 0.5], [0.45, 0.55]], (0, 2), [[0.5, 0.5], [0.45, 0.65]]),
      ([[0.5, 0.5], [0.45, 0.55]], (1, 2), [[0.5, 0.6], [0.45, 0.65]]),
      ([[0.5, 0.5], [0.45, 0.55]], (1, 3), [[0.6, 0.5], [0.55, 0.55]]),
      ([[0.5, 0.5], [0.45, 0.55]], (1, 1), [[0.4, 0.5], [0.35, 0.55]]),
      ([[0.5, 0.5], [0.45, 0.55]], (1, 0), [[0.5, 0.4], [0.45, 0.45]]),
      ([[0.95, 0.02], [0.95, 0.05]], (1, 3), [[1., 0.02], [1., 0.05]]),
      ([[0.95, 0.02], [0.95, 0.05]],
       (1, 3), [[1.05, 0.02], [1.05, 0.05]], False),
      ([[0.45, 0.55], [0.5, 0.5], [0.45, 0.55]],
       (1, 3), [[0.45, 0.55], [0.6, 0.5], [0.55, 0.55]]),
  )
  def testMoveSprites(self,
                      init_positions,
                      action,
                      final_positions,
                      keep_in_frame=True):
    """Take a series of actions and repeatedly check sprite motions."""
    action_space = action_spaces.Embodied(step_size=0.1)
    sprites = [
        sprite.Sprite(x=pos[0], y=pos[1], shape='square', scale=0.15)
        for pos in init_positions
    ]
    action_space.step(action, sprites, keep_in_frame=keep_in_frame)
    for s, p in zip(sprites, final_positions):
      self.assertTrue(np.allclose(s.position, p, atol=1e-5))


if __name__ == '__main__':
  absltest.main()
