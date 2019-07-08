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
"""Tests for environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_env import test_utils
import numpy as np
from six.moves import range

from spriteworld import action_spaces
from spriteworld import environment
from spriteworld import renderers
from spriteworld import sprite
from spriteworld import tasks


class EnvironmentTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def assertValidObservation(self, observation):
    # Override this method from test_utils.EnvironmentTestMixin to make it
    # support a dict of specs as observation.
    observation_spec = self.environment.observation_spec()
    for k, v in observation.items():
      self.assertConformsToSpec(v, observation_spec[k])

  def make_object_under_test(self):
    """Environment creator used by test_utils.EnvironmentTestMixin."""
    env = environment.Environment(
        task=tasks.NoReward(),
        action_space=action_spaces.SelectMove(),
        renderers={},
        init_sprites=lambda: [sprite.Sprite(c0=255)],
        max_episode_length=7)
    return env

  def testMaxEpisodeLength(self):
    env = self.make_object_under_test()
    action = np.array([0.5, 0.5, 0.5, 0.5])
    env.step(action)

    for _ in range(3):
      for _ in range(6):
        timestep = env.step(action)
        self.assertTrue(timestep.mid())
      timestep = env.step(action)
      self.assertTrue(timestep.last())
      timestep = env.step(action)
      self.assertTrue(timestep.first())

  def testTaskTermination(self):
    task = tasks.FindGoalPosition(goal_position=(0.5, 0.5))
    action_space = action_spaces.SelectMove()
    env_renderers = {}
    init_sprites = lambda: [sprite.Sprite(x=0.25, y=0.25, c0=255)]

    env = environment.Environment(task, action_space, env_renderers,
                                  init_sprites)
    donothing_action = np.array([0.25, 0.25, 0.5, 0.5])
    success_action = np.array([0.25, 0.25, 0.75, 0.75])

    timestep = env.step(donothing_action)
    self.assertTrue(timestep.first())

    timestep = env.step(donothing_action)
    self.assertTrue(timestep.mid())

    timestep = env.step(success_action)
    self.assertTrue(timestep.last())

    timestep = env.step(success_action)
    self.assertTrue(timestep.first())


class EnvironmentRenderersTest(absltest.TestCase):

  def make_object_under_test(self, renderer):
    self.environment = environment.Environment(
        task=tasks.NoReward(),
        action_space=action_spaces.SelectMove(),
        renderers={'obs': renderer},
        init_sprites=lambda: [sprite.Sprite(c0=255)],
        max_episode_length=7)

  def testSpriteFactors(self):
    self.make_object_under_test(renderer=renderers.SpriteFactors())
    self.environment.observation_spec()
    action = np.array([0.5, 0.5, 0.5, 0.5])
    self.environment.step(action)


if __name__ == '__main__':
  absltest.main()
