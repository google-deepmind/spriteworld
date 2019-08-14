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
r"""Goal-Finding and clustering combined task.

To demo this task, navigate to the main directory and run the following:
'''
$ python demo --config=spriteworld.configs.examples.goal_finding_clustering \
--task_hsv_colors=False
'''

This is a complicated task designed only to exemplify the features of the task
specification procedures.

In this task there are three kinds of sprites:
1) Those to be clustered. These are triangles, squares, and pentagons. They must
be clustered according to their color.
2) Those to be brought to goal regions. These are 4-spokes and 4-stars. They
must be brought to different sides of the arena according to their color.
Namely, the reddish ones must be brought to the right side of the arena and the
greenish ones must be brought to the left side of the arena (the y-position is
irrelevant).
3) Distractors. These are circles.

There is a train/test split: In test mode, the colors of the objects to be
clustered and the scales of those to be brought to goals are different.

Note that the colors in this task are defined in RGB space, so be sure when
running the demo on it to set --task_hsv_colors=False.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from spriteworld import action_spaces
from spriteworld import factor_distributions as distribs
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite_generators
from spriteworld import tasks


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: Unused task mode.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  # Factor distributions common to all objects.
  common_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Continuous('angle', 0, 360, dtype='int32'),
  ])

  # train/test split for goal-finding object scales and clustering object colors
  goal_finding_scale_test = distribs.Continuous('scale', 0.08, 0.12)
  green_blue_colors = distribs.Product([
      distribs.Continuous('c1', 64, 256, dtype='int32'),
      distribs.Continuous('c2', 64, 256, dtype='int32'),
  ])
  if mode == 'train':
    goal_finding_scale = distribs.SetMinus(
        distribs.Continuous('scale', 0.05, 0.15),
        goal_finding_scale_test,
    )
    cluster_colors = distribs.Product(
        [distribs.Continuous('c0', 128, 256, dtype='int32'), green_blue_colors])
  elif mode == 'test':
    goal_finding_scale = goal_finding_scale_test
    cluster_colors = distribs.Product(
        [distribs.Continuous('c0', 0, 128, dtype='int32'), green_blue_colors])
  else:
    raise ValueError(
        'Invalid mode {}. Mode must be "train" or "test".'.format(mode))

  # Create clustering sprite generators
  sprite_gen_list = []
  cluster_shapes = [
      distribs.Discrete('shape', [s])
      for s in ['triangle', 'square', 'pentagon']
  ]
  for shape in cluster_shapes:
    factors = distribs.Product([
        common_factors,
        cluster_colors,
        shape,
        distribs.Continuous('scale', 0.08, 0.12),
    ])
    sprite_gen_list.append(
        sprite_generators.generate_sprites(factors, num_sprites=2))

  # Create goal-finding sprite generators
  goal_finding_colors = [
      distribs.Product([
          distribs.Continuous('c0', 192, 256, dtype='int32'),
          distribs.Continuous('c1', 0, 128, dtype='int32'),
          distribs.Continuous('c2', 64, 128, dtype='int32'),
      ]),
      distribs.Product([
          distribs.Continuous('c0', 0, 128, dtype='int32'),
          distribs.Continuous('c1', 192, 256, dtype='int32'),
          distribs.Continuous('c2', 64, 128, dtype='int32'),
      ])
  ]
  # Goal positions corresponding to the colors in goal_finding_colors
  goal_finding_positions = [(0., 0.5), (1., 0.5)]
  goal_finding_shapes = distribs.Discrete('shape', ['spoke_4', 'star_4'])
  for colors in goal_finding_colors:
    factors = distribs.Product([
        common_factors,
        goal_finding_scale,
        goal_finding_shapes,
        colors,
    ])
    sprite_gen_list.append(
        sprite_generators.generate_sprites(
            factors, num_sprites=lambda: np.random.randint(1, 3)))

  # Create distractor sprite generator
  distractor_factors = distribs.Product([
      common_factors,
      distribs.Discrete('shape', ['circle']),
      distribs.Continuous('c0', 64, 256, dtype='uint8'),
      distribs.Continuous('c1', 64, 256, dtype='uint8'),
      distribs.Continuous('c2', 64, 256, dtype='uint8'),
      distribs.Continuous('scale', 0.08, 0.12),
  ])
  sprite_gen_list.append(sprite_generators.generate_sprites(
      distractor_factors, num_sprites=lambda: np.random.randint(0, 3)))

  # Concat clusters into single scene to generate
  sprite_gen = sprite_generators.chain_generators(*sprite_gen_list)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  # Create the combined task of goal-finding and clustering
  task_list = []
  task_list.append(
      tasks.Clustering(cluster_shapes, terminate_bonus=0., reward_range=10.))
  for colors, goal_pos in zip(goal_finding_colors, goal_finding_positions):
    goal_finding_task = tasks.FindGoalPosition(
        distribs.Product([colors, goal_finding_shapes]),
        goal_position=goal_pos,
        weights_dimensions=(1, 0),
        terminate_distance=0.15,
        raw_reward_multiplier=30)
    task_list.append(goal_finding_task)
  task = tasks.MetaAggregated(
      task_list, reward_aggregator='sum', termination_criterion='all')

  renderers = {
      'image':
          spriteworld_renderers.PILRenderer(
              image_size=(64, 64), anti_aliasing=5)
  }

  config = {
      'task': task,
      'action_space': action_spaces.SelectMove(scale=0.5),
      'renderers': renderers,
      'init_sprites': sprite_gen,
      'max_episode_length': 50,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }
  return config
