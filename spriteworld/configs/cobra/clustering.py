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
"""Clustering task used in COBRA.

Cluster sprites by color.

We use 4 types of sprites, based on their hue.
We then compute a Davies-Bouldin clustering metric to assess clustering quality
(and generate a reward). The Clustering task uses a threshold to terminate an
episode when the clustering metric is good enough.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld.configs.cobra import common

# Task Parameters
NUM_SPRITES_PER_CLUSTER = 2
MAX_EPISODE_LENGTH = 50

# Define possible clusters (here using Hue as selection attribute)
CLUSTERS_DISTS = {
    'red': distribs.Continuous('c0', 0.9, 1.),
    'blue': distribs.Continuous('c0', 0.55, 0.65),
    'green': distribs.Continuous('c0', 0.27, 0.37),
    'yellow': distribs.Continuous('c0', 0.1, 0.2),
}

# Define train/test generalization splits
MODES = {
    'train': ('blue', 'green'),
    'test': ('red', 'yellow'),
}


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  # Select clusters to use, and their c0 factor distribution.
  c0_clusters = [CLUSTERS_DISTS[cluster] for cluster in MODES[mode]]
  print('Clustering task: {}, #sprites: {}'.format(MODES[mode],
                                                   NUM_SPRITES_PER_CLUSTER))

  other_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('shape', ['square', 'triangle', 'circle']),
      distribs.Discrete('scale', [0.13]),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
  ])

  # Generate the sprites to be used in this task, by combining Hue with the
  # other factors.
  sprite_factors = [
      distribs.Product((other_factors, c0)) for c0 in c0_clusters
  ]
  # Convert to sprites, generating the appropriate number per cluster.
  sprite_gen_per_cluster = [
      sprite_generators.generate_sprites(
          factors, num_sprites=NUM_SPRITES_PER_CLUSTER)
      for factors in sprite_factors
  ]
  # Concat clusters into single scene to generate.
  sprite_gen = sprite_generators.chain_generators(*sprite_gen_per_cluster)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  # Clustering task will define rewards
  task = tasks.Clustering(c0_clusters, terminate_bonus=0., reward_range=10.)

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
