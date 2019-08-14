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
"""Goal-Finding tasks testing for generalization to new initial positions.

In this task there is one target sprite of orange-green-ish color and one
distractor sprite of blue-purple-ish color. The target must be brought to the
goal location, which is the center of the arena, while the distractor does not
contribute to the reward.

In train mode the target is initialized in any position except the lower-right
quadrant, while in test mode it is initialized only in the lower-right quadrant.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld.configs.cobra import common

TERMINATE_DISTANCE = 0.075
NUM_TARGETS = 1
NUM_DISTRACTORS = 1
MODES_TARGET_POSITIONS = {
    'train':
        distribs.SetMinus(
            distribs.Product((
                distribs.Continuous('x', 0.1, 0.9),
                distribs.Continuous('y', 0.1, 0.9),
            )),
            distribs.Product((
                distribs.Continuous('x', 0.5, 0.9),
                distribs.Continuous('y', 0.5, 0.9),
            )),
        ),
    'test':
        distribs.Product((
            distribs.Continuous('x', 0.5, 0.9),
            distribs.Continuous('y', 0.5, 0.9),
        )),
}


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  shared_factors = distribs.Product([
      distribs.Discrete('shape', ['square', 'triangle', 'circle']),
      distribs.Discrete('scale', [0.13]),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
  ])
  target_hue = distribs.Continuous('c0', 0., 0.4)
  distractor_hue = distribs.Continuous('c0', 0.5, 0.9)
  target_factors = distribs.Product([
      MODES_TARGET_POSITIONS[mode],
      target_hue,
      shared_factors,
  ])
  distractor_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distractor_hue,
      shared_factors,
  ])

  target_sprite_gen = sprite_generators.generate_sprites(
      target_factors, num_sprites=NUM_TARGETS)
  distractor_sprite_gen = sprite_generators.generate_sprites(
      distractor_factors, num_sprites=NUM_DISTRACTORS)
  sprite_gen = sprite_generators.chain_generators(target_sprite_gen,
                                                  distractor_sprite_gen)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  task = tasks.FindGoalPosition(
      filter_distrib=target_hue, terminate_distance=TERMINATE_DISTANCE)

  config = {
      'task': task,
      'action_space': common.action_space(),
      'renderers': common.renderers(),
      'init_sprites': sprite_gen,
      'max_episode_length': 20,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }
  return config
