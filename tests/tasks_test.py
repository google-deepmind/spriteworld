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
"""Tests for tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np

from spriteworld import factor_distributions as distribs
from spriteworld import sprite
from spriteworld import tasks


class GoalPositionTest(parameterized.TestCase):

  def _mock_sprites(self, sprite_positions):
    sprites = []
    for sprite_pos in sprite_positions:
      mocksprite = mock.Mock(spec=sprite.Sprite)
      mocksprite.position = sprite_pos
      sprites.append(mocksprite)
    return sprites

  @parameterized.parameters(
      ([np.array([0., 0.])], (0.5, 0.5), -30.4, False),
      ([np.array([0.4, 0.6])], (0.5, 0.5), -2.1, False),
      ([np.array([0.43, 0.56])], (0.5, 0.5), 0.4, True),
      ([np.array([0.48, 0.52]), np.array([0.4, 0.6])], (0.5, 0.5), 1.5, False),
      ([np.array([0.48, 0.52]), np.array([0.5, 0.5])], (0.5, 0.5), 8.6, True))
  def testBasicReward(self, sprite_positions, goal_position, reward, success):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=goal_position, terminate_distance=0.1)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)
    self.assertEqual(task.success(sprites), success)

  @parameterized.parameters(
      ([np.array([0.4, 0.6])], 0.15, 0.4, True),
      ([np.array([0.36, 0.5])], 0.15, 0.5, True),
      ([np.array([0.34, 0.5])], 0.15, -0.5, False),
      ([np.array([0.34, 0.5])], 0.2, 2., True),
      ([np.array([0.34, 0.39])], 0.2, 0.2, True),
      ([np.array([0.34, 0.37])], 0.2, -0.3, False))
  def testTerminateDistance(self, sprite_positions, terminate_distance, reward,
                            success):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=(0.5, 0.5), terminate_distance=terminate_distance)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)
    self.assertEqual(task.success(sprites), success)

  @parameterized.parameters(
      ([np.array([0.4, 0.52])], 3., -0.1),
      ([np.array([0.43, 0.52])], 3., 4.4),
      ([np.array([0.43, 0.52])], 1., 2.4),
      ([np.array([0.43, 0.52]), np.array([0.4, 0.52])], 3., 1.3),
      ([np.array([0.43, 0.52]), np.array([0.43, 0.52])], 3., 5.7))
  def testTerminateBonus(self, sprite_positions, terminate_bonus, reward):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=(0.5, 0.5),
        terminate_distance=0.1,
        terminate_bonus=terminate_bonus)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)

  @parameterized.parameters(
      ([np.array([0.43, 0.52])], (1, 1), 1.4, True),
      ([np.array([0.43, 0.52])], (3, 1), -1.1, False),
      ([np.array([0.3, 0.52])], (7, 2), -21.5, False),
      ([np.array([0.3, 0.52])], (0.1, 0.2), 1.8, True),
  )
  def testWeightsDimensions(self, sprite_positions, weights_dimensions, reward,
                            success):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=(0.5, 0.5),
        terminate_distance=0.1,
        weights_dimensions=weights_dimensions)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)

    self.assertEqual(task.success(sprites), success)

  @parameterized.parameters(
      ([np.array([0.35, 0.52])], 0., 50.0, -2.6),
      ([np.array([0.35, 0.52])], 0., 10.0, -0.5),
      ([np.array([0.43, 0.52])], 1., 10.0, 1.3),
      ([np.array([0.43, 0.52]), np.array([0.4, 0.52])], 0., 50.0, 1.3),
      ([np.array([0.43, 0.52]), np.array([0.43, 0.52])], 0., 10.0, 0.5))
  def testRewardMultiplier(self, sprite_positions, terminate_bonus,
                           reward_multiplier, reward):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=(0.5, 0.5),
        terminate_distance=0.1,
        terminate_bonus=terminate_bonus,
        raw_reward_multiplier=reward_multiplier)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)

  @parameterized.parameters(
      ([np.array([0.35, 0.52])], 1., 0.),
      ([np.array([0.43, 0.52])], 1., 2.4),
      ([np.array([0.43, 0.52])], 3., 4.4),
      ([np.array([0.43, 0.52]), np.array([0.4, 0.55])], 1., 0.),
      ([np.array([0.43, 0.52]), np.array([0.43, 0.52])], 1., 3.7),
      ([np.array([0.43, 0.52]), np.array([0.43, 0.52])], 3., 5.7))
  def testSparseReward(self, sprite_positions, terminate_bonus, reward):
    sprites = self._mock_sprites(sprite_positions)
    task = tasks.FindGoalPosition(
        goal_position=(0.5, 0.5),
        terminate_distance=0.1,
        sparse_reward=True,
        terminate_bonus=terminate_bonus)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)

  def testFilterDistrib(self):
    sprites = [
        sprite.Sprite(x=0.45, y=0.45, c0=64),
        sprite.Sprite(x=0.45, y=0.55, c0=128),
        sprite.Sprite(x=0.55, y=0.45, c0=192),
        sprite.Sprite(x=0.4, y=0.4, c0=255),
    ]

    filter_distribs = [
        distribs.Continuous('c0', 0, 65),  # selects sprites[:1]
        distribs.Continuous('c0', 0, 129),  # selects sprites[:2]
        distribs.Continuous('c0', 0, 193),  # selects sprites[:3]
        distribs.Continuous('c0', 0, 256),  # selects sprites[:4]
        distribs.Continuous('c0', 65, 256),  # selects sprites[1:4]
    ]

    task_list = [
        tasks.FindGoalPosition(
            filter_distrib=x, goal_position=(0.5, 0.5), terminate_distance=0.1)
        for x in filter_distribs
    ]

    rewards = [1.5, 2.9, 4.4, 2.3, 0.9]
    successes = [True, True, True, False, False]
    for t, r, s in zip(task_list, rewards, successes):
      self.assertAlmostEqual(t.reward(sprites), r, delta=0.1)
      self.assertEqual(t.success(sprites), s)

  def testNoFilteredSprites(self):
    sprites = [sprite.Sprite(x=0.45, y=0.45, c0=255)]
    filter_distrib = distribs.Continuous('c0', 0, 254)
    r = tasks.FindGoalPosition(
        filter_distrib=filter_distrib,
        goal_position=(0.5, 0.5),
        terminate_distance=0.1).reward(sprites)
    self.assertTrue(np.isnan(r))


class ClusteringTest(parameterized.TestCase):

  def setUp(self):
    super(ClusteringTest, self).setUp()
    self.sprites = [
        sprite.Sprite(x=0.2, y=0.2, c0=64),
        sprite.Sprite(x=0.3, y=0.3, c0=128),
        sprite.Sprite(x=0.8, y=0.9, c0=192),
        sprite.Sprite(x=0.9, y=0.8, c0=255),
    ]
    self.cluster_distribs = [
        distribs.Continuous('c0', 0, 129),
        distribs.Continuous('c0', 190, 256),
    ]

  @parameterized.parameters(
      ([[0.2, 0.2], [0.21, 0.21], [0.8, 0.8], [0.81, 0.81]], 287.5
       , True),
      ([[0.2, 0.2], [0.25, 0.25], [0.8, 0.8], [0.81, 0.81]], 84.2, True),
      ([[0.2, 0.2], [0.53, 0.53], [0.8, 0.8], [0.81, 0.81]], 0.4, True),
      ([[0.2, 0.53], [0.53, 0.2], [0.8, 0.8], [0.81, 0.81]], 0.4, True),
      ([[0.2, 0.2], [0.53, 0.53], [0.8, 0.8], [0.9, 0.9]], -1.2, False),
      ([[0.2, 0.2], [0.53, 0.53], [0.8, 0.9], [0.9, 0.8]], -1.2, False),
  )
  def test4Sprites(self, positions, reward, success):
    sprites = [
        sprite.Sprite(x=positions[0][0], y=positions[0][1], c0=64),
        sprite.Sprite(x=positions[1][0], y=positions[1][1], c0=128),
        sprite.Sprite(x=positions[2][0], y=positions[2][1], c0=192),
        sprite.Sprite(x=positions[3][0], y=positions[3][1], c0=255),
    ]
    cluster_distribs = [
        distribs.Continuous('c0', 0, 129),
        distribs.Continuous('c0', 190, 256),
    ]
    task = tasks.Clustering(cluster_distribs=cluster_distribs)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)
    self.assertEqual(task.success(sprites), success)

  @parameterized.parameters(
      ([[0.2, 0.2], [0.3, 0.3]], [[0.8, 0.8], [0.8, 0.9], [0.9, 0.9]], 18.7),
      ([[0.2, 0.2], [0.3, 0.3]], [[0.8, 0.8], [0.8, 0.9], [0.9, 0.2]], -2.9),
      ([[0.2, 0.2], [0.3, 0.3], [0.25, 0.3]], [[0.8, 0.8], [0.8, 0.9],
                                               [0.9, 0.9]], 21.2),
      ([[0.2, 0.2], [0.3, 0.3], [0.4, 0.8]], [[0.8, 0.8], [0.8, 0.9],
                                              [0.9, 0.9]], -1.8),
  )
  def testMoreSprites(self, positions_0, positions_1, reward):
    sprites = [sprite.Sprite(x=p[0], y=p[1], c0=75) for p in positions_0]
    sprites.extend([sprite.Sprite(x=p[0], y=p[1], c0=225) for p in positions_1])
    cluster_distribs = [
        distribs.Continuous('c0', 50, 100),
        distribs.Continuous('c0', 200, 250),
    ]
    task = tasks.Clustering(cluster_distribs=cluster_distribs)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)

  def test3Clusters(self):
    sprites = [
        sprite.Sprite(x=0.2, y=0.2, c0=64),
        sprite.Sprite(x=0.3, y=0.3, c0=64),
        sprite.Sprite(x=0.8, y=0.9, c0=128),
        sprite.Sprite(x=0.9, y=0.8, c0=128),
        sprite.Sprite(x=0.8, y=0.9, c0=255),
        sprite.Sprite(x=0.9, y=0.8, c0=255),
    ]
    cluster_distribs = [
        distribs.Continuous('c0', 0, 100),
        distribs.Continuous('c0', 100, 150),
        distribs.Continuous('c0', 200, 256),
    ]
    task = tasks.Clustering(cluster_distribs=cluster_distribs)
    self.assertAlmostEqual(task.reward(sprites), 17.5, delta=0.1)

  @parameterized.parameters(
      (2.5, 17.5),
      (5., 5.),
      (10., -20.),
  )
  def testTerminationThreshold(self, termination_threshold, reward):
    task = tasks.Clustering(
        cluster_distribs=self.cluster_distribs,
        termination_threshold=termination_threshold)
    self.assertAlmostEqual(task.reward(self.sprites), reward, delta=0.1)

  @parameterized.parameters(
      (2.5, 0., 17.5),
      (2.5, 5., 22.5),
      (5., 3., 8.),
      (10., 7., -20.),
  )
  def testTerminateBonus(self, termination_threshold, terminate_bonus, reward):
    task = tasks.Clustering(
        cluster_distribs=self.cluster_distribs,
        terminate_bonus=terminate_bonus,
        termination_threshold=termination_threshold)
    self.assertAlmostEqual(task.reward(self.sprites), reward, delta=0.1)

  @parameterized.parameters(
      (2.5, 10., 17.5),
      (2.5, 5., 8.8),
      (5., 3., 1.5),
      (10., 7., -14.),
  )
  def testRewardRange(self, termination_threshold, reward_range, reward):
    task = tasks.Clustering(
        cluster_distribs=self.cluster_distribs,
        reward_range=reward_range,
        termination_threshold=termination_threshold)
    self.assertAlmostEqual(task.reward(self.sprites), reward, delta=0.1)

  @parameterized.parameters(
      (2.5, 0., 17.5),
      (7., 0., 0.),
      (10., 0., 0.),
      (5., 5., 10.),
  )
  def testSparseReward(self, termination_threshold, terminate_bonus, reward):
    task = tasks.Clustering(
        cluster_distribs=self.cluster_distribs,
        sparse_reward=True,
        terminate_bonus=terminate_bonus,
        termination_threshold=termination_threshold)
    self.assertAlmostEqual(task.reward(self.sprites), reward, delta=0.1)


class MetaAggregatedTest(parameterized.TestCase):

  def setUp(self):
    super(MetaAggregatedTest, self).setUp()
    self.subtasks = [
        tasks.FindGoalPosition(
            filter_distrib=distribs.Continuous('c0', 0, 100),
            goal_position=np.array([0.2, 0.2]),
            terminate_distance=0.1),
        tasks.FindGoalPosition(
            filter_distrib=distribs.Continuous('c0', 100, 200),
            goal_position=np.array([0.5, 0.5]),
            terminate_distance=0.1,
            terminate_bonus=5.0),
        tasks.FindGoalPosition(
            filter_distrib=distribs.Continuous('c0', 200, 256),
            goal_position=np.array([0.8, 0.8]),
            terminate_distance=0.1,
            terminate_bonus=10.0),
    ]
    self.success_sprites = [
        sprite.Sprite(x=0.2, y=0.2, c0=50),
        sprite.Sprite(x=0.5, y=0.45, c0=150),
        sprite.Sprite(x=0.85, y=0.75, c0=250),
    ]
    self.success_rewards = [5., 7.5, 11.5]
    self.failure_sprites = [
        sprite.Sprite(x=0.2, y=0.8, c0=50),
        sprite.Sprite(x=0.3, y=0.45, c0=150),
        sprite.Sprite(x=0.9, y=0.75, c0=250),
    ]
    self.failure_rewards = [-25., -5.3, -0.6]

  def _get_sprites_and_reward_list(self, successes):
    success_inds = np.nonzero(successes)[0]
    failure_inds = np.nonzero(np.logical_not(successes))[0]
    sprites = [self.success_sprites[i] for i in success_inds]
    sprites.extend([self.failure_sprites[i] for i in failure_inds])
    reward_list = [self.success_rewards[i] for i in success_inds]
    reward_list.extend([self.failure_rewards[i] for i in failure_inds])
    return sprites, reward_list

  @parameterized.parameters(
      ('all', (True, True, True), True),
      ('all', (True, True, False), False),
      ('all', (True, False, False), False),
      ('all', (False, False, False), False),
      ('any', (True, True, True), True),
      ('any', (True, True, False), True),
      ('any', (True, False, False), True),
      ('any', (False, False, False), False),
  )
  def testSum(self, termination_criterion, successes, success):
    task = tasks.MetaAggregated(
        self.subtasks,
        reward_aggregator='sum',
        termination_criterion=termination_criterion)
    sprites, reward_list = self._get_sprites_and_reward_list(successes)
    self.assertAlmostEqual(task.reward(sprites), sum(reward_list), delta=0.1)
    self.assertEqual(task.success(sprites), success)

  @parameterized.parameters(
      ((True, True, True),),
      ((True, True, False),),
      ((True, False, False),),
      ((False, False, False),),
  )
  def testMax(self, successes):
    task = tasks.MetaAggregated(
        self.subtasks,
        reward_aggregator='max')
    sprites, reward_list = self._get_sprites_and_reward_list(successes)
    self.assertAlmostEqual(task.reward(sprites), max(reward_list), delta=0.1)

  @parameterized.parameters(
      ((True, True, True),),
      ((True, True, False),),
      ((True, False, False),),
      ((False, False, False),),
  )
  def testMin(self, successes):
    task = tasks.MetaAggregated(self.subtasks, reward_aggregator='min')
    sprites, reward_list = self._get_sprites_and_reward_list(successes)
    self.assertAlmostEqual(task.reward(sprites), min(reward_list), delta=0.1)

  @parameterized.parameters(
      ((True, True, True),),
      ((True, True, False),),
      ((True, False, False),),
      ((False, False, False),),
  )
  def testMean(self, successes):
    task = tasks.MetaAggregated(self.subtasks, reward_aggregator='mean')
    sprites, reward_list = self._get_sprites_and_reward_list(successes)
    self.assertAlmostEqual(
        task.reward(sprites), np.mean(reward_list), delta=0.1)

  @parameterized.parameters(
      ((True, True, True), 'sum', 0., 24.),
      ((True, True, True), 'sum', 5., 29.),
      ((True, True, False), 'sum', 5., 11.9),
      ((True, True, True), 'min', 0., 5.),
      ((True, True, True), 'min', 5., 10.),
      ((True, True, False), 'min', 5., -0.6),
  )
  def testTerminateBonus(self, successes, reward_aggregator, terminate_bonus,
                         reward):
    task = tasks.MetaAggregated(
        self.subtasks,
        reward_aggregator=reward_aggregator,
        terminate_bonus=terminate_bonus)
    sprites, _ = self._get_sprites_and_reward_list(successes)
    self.assertAlmostEqual(task.reward(sprites), reward, delta=0.1)


if __name__ == '__main__':
  absltest.main()
