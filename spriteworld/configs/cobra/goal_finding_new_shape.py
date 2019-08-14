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
"""Goal-Finding tasks testing for generalization to new shapes.

In this task there is one sprite per episode. That sprite must be brought to the
goal location, which is always the center of the arena. At training time the
sprite is a square. At test time it is either a circle or a triangle.
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
MODES_SHAPES = {
    'train': distribs.Discrete('shape', ['square']),
    'test': distribs.Discrete('shape', ['triangle', 'circle']),
}


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  factors = distribs.Product([
      MODES_SHAPES[mode],
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('scale', [0.13]),
      distribs.Continuous('c0', 0., 0.4),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
  ])
  sprite_gen = sprite_generators.generate_sprites(
      factors, num_sprites=NUM_TARGETS)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  task = tasks.FindGoalPosition(terminate_distance=TERMINATE_DISTANCE)

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
