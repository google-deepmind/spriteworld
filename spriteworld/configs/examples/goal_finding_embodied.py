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
"""Goal-Finding task for embodied agent.

In this task there are target sprites of orange-green-ish color. All target
sprites must be brought to the goal location, which is the center of the arena.
There are also distractor sprites, which are blue-purple-ish color and do not
contribute to the reward.
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

TERMINATE_DISTANCE = 0.075
NUM_TARGETS = lambda: np.random.randint(1, 4)
NUM_DISTRACTORS = lambda: np.random.randint(1, 4)


def get_config(mode=None):
  """Generate environment config.

  Args:
    mode: Unused task mode.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """
  del mode

  shared_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('shape', ['square', 'triangle', 'circle']),
      distribs.Discrete('scale', [0.13]),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
  ])
  target_hue = distribs.Continuous('c0', 0., 0.4)
  distractor_hue = distribs.Continuous('c0', 0.5, 0.9)
  target_factors = distribs.Product([
      target_hue,
      shared_factors,
  ])
  distractor_factors = distribs.Product([
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

  # Create the agent body
  agent_body_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('shape', ['circle']),
      distribs.Discrete('scale', [0.07]),
      distribs.Discrete('c0', [1.]),
      distribs.Discrete('c1', [0.]),
      distribs.Discrete('c2', [1.]),
  ])
  agent_body_gen = sprite_generators.generate_sprites(
      agent_body_factors, num_sprites=1)
  sprite_gen = sprite_generators.chain_generators(sprite_gen, agent_body_gen)

  task = tasks.FindGoalPosition(
      filter_distrib=target_hue, terminate_distance=TERMINATE_DISTANCE)

  renderers = {
      'image':
          spriteworld_renderers.PILRenderer(
              image_size=(64, 64),
              anti_aliasing=5,
              color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb)
  }

  config = {
      'task': task,
      'action_space': action_spaces.Embodied(step_size=0.05),
      'renderers': renderers,
      'init_sprites': sprite_gen,
      'max_episode_length': 50,
      'metadata': {
          'name': os.path.basename(__file__),
      }
  }
  return config
