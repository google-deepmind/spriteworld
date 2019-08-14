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
"""Sorting task used in COBRA.

Sort sprites into target locations based on color.

We use 5 narrow hue ranges (red, blue, green, purple, yellow), and associated to
each a goal location (the corners and center of the arena). Each episode we
sample two sprites in random locations with different colors and reward the
agent for bringing them each to their respective goal location. in the training
mode we hold out one color pair, and in the test mode we sample only that pair.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import numpy as np
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld.configs.cobra import common

# Task Parameters
MAX_EPISODE_LENGTH = 50
TERMINATE_DISTANCE = 0.075
RAW_REWARD_MULTIPLIER = 20.
NUM_TARGETS = 2

# Sub-tasks for each color/goal
SUBTASKS = (
    {
        'distrib': distribs.Continuous('c0', 0.9, 1.),  # red
        'goal_position': np.array([0.75, 0.75])
    },
    {
        'distrib': distribs.Continuous('c0', 0.55, 0.65),  # blue
        'goal_position': np.array([0.75, 0.25])
    },
    {
        'distrib': distribs.Continuous('c0', 0.27, 0.37),  # green
        'goal_position': np.array([0.25, 0.75])
    },
    {
        'distrib': distribs.Continuous('c0', 0.73, 0.83),  # purple
        'goal_position': np.array([0.25, 0.25])
    },
    {
        'distrib': distribs.Continuous('c0', 0.1, 0.2),  # yellow
        'goal_position': np.array([0.5, 0.5])
    },
)


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  # Create the subtasks and their corresponding sprite generators
  subtasks = []
  sprite_gen_per_subtask = []
  for subtask in SUBTASKS:
    subtasks.append(tasks.FindGoalPosition(
        filter_distrib=subtask['distrib'],
        goal_position=subtask['goal_position'],
        terminate_distance=TERMINATE_DISTANCE,
        raw_reward_multiplier=RAW_REWARD_MULTIPLIER))
    factors = distribs.Product((
        subtask['distrib'],
        distribs.Continuous('x', 0.1, 0.9),
        distribs.Continuous('y', 0.1, 0.9),
        distribs.Discrete('shape', ['square', 'triangle', 'circle']),
        distribs.Discrete('scale', [0.13]),
        distribs.Continuous('c1', 0.3, 1.),
        distribs.Continuous('c2', 0.9, 1.),
        ))
    sprite_gen_per_subtask.append(
        sprite_generators.generate_sprites(factors, num_sprites=1))

  # Consider all combinations of subtasks
  subtask_combos = list(
      itertools.combinations(np.arange(len(SUBTASKS)), NUM_TARGETS))
  if mode == 'train':
    # Randomly sample a combination of subtasks, holding one combination out
    sprite_gen = sprite_generators.sample_generator([
        sprite_generators.chain_generators(
            *[sprite_gen_per_subtask[i] for i in c]) for c in subtask_combos[1:]
    ])

  elif mode == 'test':
    # Use the held-out subtask combination for testing
    sprite_gen = sprite_generators.chain_generators(
        *[sprite_gen_per_subtask[i] for i in subtask_combos[0]])
  else:
    raise ValueError('Invalide mode {}.'.format(mode))

  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  task = tasks.MetaAggregated(
      subtasks, reward_aggregator='sum', termination_criterion='all')

  config = {
      'task': task,
      'action_space': common.action_space(),
      'renderers': common.renderers(),
      'init_sprites': sprite_gen,
      'max_episode_length': MAX_EPISODE_LENGTH,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }

  return config
