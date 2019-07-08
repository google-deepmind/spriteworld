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
"""Handcrafted renderers for Spriteworld."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np
from spriteworld import constants
from spriteworld import sprite as sprite_lib
from spriteworld.renderers import abstract_renderer


class SpriteFactors(abstract_renderer.AbstractRenderer):
  """Aggregates factors of the sprites into an array."""

  def __init__(self, factors=sprite_lib.FACTOR_NAMES):
    """Constructor.

    Outputs a list of dicts: [{object 1 factors} {object 2 factors} ...]

    Args:
      factors: Iterable of strings. Factors to record. Must be a subset of
        sprite.FACTOR_NAMES.
    """
    if not set(factors).issubset(set(sprite_lib.FACTOR_NAMES)):
      raise ValueError('Factors have to belong to {}.'.format(
          sprite_lib.FACTOR_NAMES))
    self._num_sprites = None
    self._factors = factors

    self._per_object_spec = {
        factor: specs.Array(shape=(), dtype=np.float32) for factor in factors
    }

  def render(self, sprites=(), global_state=None):
    """Renders a list of sprites into a list of sprite factors.

    Args:
      sprites: a list of sprites with a method `get_sprite`. This method
        receives a single argument `upscale_factor`, and returns a pygame
        sprite.
      global_state: Unused global state.

    Returns:
      A list of dictionaries of factor -> values mappings.
    """
    del global_state

    # Set number of sprites so that observation_spec is callable
    self._num_sprites = len(sprites)

    def _process_factor(name, value):
      if name == 'shape':
        value = constants.ShapeType[value].value
      return float(value)

    def _sprite_to_factors(sprite):
      return {
          factor: _process_factor(factor, getattr(sprite, factor))
          for factor in self._factors
      }

    return np.array([_sprite_to_factors(sprite) for sprite in sprites])

  def observation_spec(self):
    return [self._per_object_spec for _ in range(self._num_sprites)]


class SpritePassthrough(abstract_renderer.AbstractRenderer):
  """Passes the list of Sprites directly as observation."""

  def __init__(self):
    """Constructor."""
    self._num_sprites = None

  def render(self, sprites=(), global_state=None):
    """Sends the sprites (e.g. list of Sprites) directly through.

    Args:
      sprites: a list of sprites with a method `get_sprite`. This method
        receives a single argument `upscale_factor`, and returns a pygame
        sprite.
      global_state: Unused global state.

    Returns:
      A numpy array containing the concatenation of all desired attributes of
        all sprites.
    """
    del global_state

    self._num_sprites = len(sprites)

    return sprites

  def observation_spec(self):
    return specs.Array(shape=(self._num_sprites,), dtype=np.object)


class Success(abstract_renderer.AbstractRenderer):
  """Renders whether a task has been successfully solved."""

  def render(self, sprites=(), global_state=None):
    """Returns task success.

    Args:
      sprites: Unused iterable of sprites.
      global_state: Must be a dictionary with key 'success'.

    Returns:
      Boolean indicating success.
    """
    return global_state['success']

  def observation_spec(self):
    return specs.Array(shape=(), dtype=np.bool)
