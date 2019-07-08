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
"""Tests for gym wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from gym import spaces
import numpy as np
from six.moves import range

from spriteworld import action_spaces
from spriteworld import environment
from spriteworld import gym_wrapper
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite
from spriteworld import tasks


class GymWrapperTest(absltest.TestCase):

  def testContinuousActions(self):
    renderers = {
        'image': spriteworld_renderers.PILRenderer(image_size=(64, 64),)
    }
    init_sprites = lambda: [sprite.Sprite(c0=255)]
    max_episode_length = 5
    spriteworld_env = environment.Environment(
        tasks.NoReward(),
        action_spaces.SelectMove(),
        renderers,
        init_sprites,
        max_episode_length=max_episode_length)
    env = gym_wrapper.GymWrapper(spriteworld_env)

    self.assertEqual(
        env.observation_space,
        spaces.Dict({
            'image':
                spaces.Box(-np.inf, np.inf, shape=(64, 64, 3), dtype=np.uint8)
        }))
    self.assertEqual(env.action_space,
                     spaces.Box(0., 1., shape=(4,), dtype=np.float32))

    for _ in range(3):
      env.reset()
      for _ in range(max_episode_length - 1):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        self.assertEqual(obs['image'].dtype, np.uint8)
        self.assertFalse(done)
        self.assertEqual(reward, 0.)
      action = env.action_space.sample()
      _, _, done, _ = env.step(action)
      self.assertTrue(done)
      _, _, done, _ = env.step(action)
      self.assertFalse(done)

  def testEmbodiedActions(self):
    renderers = {
        'image': spriteworld_renderers.PILRenderer(image_size=(64, 64),)
    }
    init_sprites = lambda: [sprite.Sprite(c0=255)]
    max_episode_length = 5
    spriteworld_env = environment.Environment(
        tasks.NoReward(),
        action_spaces.Embodied(),
        renderers,
        init_sprites,
        max_episode_length=max_episode_length)
    env = gym_wrapper.GymWrapper(spriteworld_env)

    self.assertEqual(
        env.observation_space,
        spaces.Dict({
            'image':
                spaces.Box(-np.inf, np.inf, shape=(64, 64, 3), dtype=np.uint8)
        }))
    self.assertEqual(env.action_space,
                     spaces.Tuple([spaces.Discrete(2),
                                   spaces.Discrete(4)]))

    for _ in range(3):
      env.reset()
      for _ in range(max_episode_length - 1):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        self.assertEqual(obs['image'].dtype, np.uint8)
        self.assertFalse(done)
        self.assertEqual(reward, 0.)
      action = env.action_space.sample()
      _, _, done, _ = env.step(action)
      self.assertTrue(done)
      _, _, done, _ = env.step(action)
      self.assertFalse(done)


if __name__ == '__main__':
  absltest.main()
