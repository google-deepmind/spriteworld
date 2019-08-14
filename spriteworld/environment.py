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
"""Spriteworld environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
import numpy as np
import six


class Environment(dm_env.Environment):
  """Environment class for Spriteworld.

  This environment uses the `dm_env` interface. For details, see
  https://github.com/deepmind/dm_env
  """

  def __init__(self,
               task,
               action_space,
               renderers,
               init_sprites,
               keep_in_frame=True,
               max_episode_length=1000,
               metadata=None):
    """Construct Spriteworld environment.

    Args:
      task: Object with methods:
          - reward: sprites -> float.
          - success: sprites -> bool.
      action_space: Action space with methods:
          - step: action, sprites, keep_in_frame -> reward.
          - action_spec: Callable returning ArraySpec or list/dict of such.
      renderers: Dict where values are renderers and keys are names, reflected
        in the keys of the observation.
      init_sprites: Callable returning iterable of sprites, called upon
        environment reset.
      keep_in_frame: Bool. Whether to keep sprites in frame when they move. This
        prevents episodes from terminating frequently when an agent moves a
        sprite out of frame.
      max_episode_length: Maximum number of steps beyond which episode will be
        terminated.
      metadata: Optional object to be added to the global_state.
    """
    self._task = task
    self._action_space = action_space
    self._renderers = renderers
    self._init_sprites = init_sprites
    self._keep_in_frame = keep_in_frame
    self._max_episode_length = max_episode_length
    self._sprites = self._init_sprites()
    self._step_count = 0
    self._reset_next_step = True
    self._renderers_initialized = False
    self._metadata = metadata

  def reset(self):
    self._sprites = self._init_sprites()
    self._step_count = 0
    self._reset_next_step = False
    return dm_env.restart(self.observation())

  def success(self):
    return self._task.success(self._sprites)

  def should_terminate(self):
    timeout = self._step_count >= self._max_episode_length
    out_of_frame = any([sprite.out_of_frame for sprite in self._sprites])
    return self.success() or out_of_frame or timeout

  def step(self, action):
    """Step the environment with an action."""
    if self._reset_next_step:
      return self.reset()

    self._step_count += 1
    reward = self._action_space.step(
        action, self._sprites, keep_in_frame=self._keep_in_frame)

    # Update sprite positions from their velocities
    for sprite in self._sprites:
      sprite.update_position(keep_in_frame=self._keep_in_frame)

    reward += self._task.reward(self._sprites)
    observation = self.observation()

    if self.should_terminate():
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=observation)
    else:
      return dm_env.transition(reward=reward, observation=observation)

  def sample_contained_position(self):
    """Sample a random position contained in a sprite.

    This is useful for hand-crafted random agents.

    Note that this function does not uniformly sample with respect to sprite
    areas. Instead, it randomly selects a sprite, then selects a random position
    within that sprite. Consequently, small sprites are represented equally to
    large sprites, and in the case of occlusion forground sprites may be
    overrepresented relative to background sprites.

    Returns:
      Float numpy array of shape (2,) in [0, 1]. Position contained in one of
          the sprites.
    """
    sprite = self._sprites[np.random.randint(len(self._sprites))]
    return sprite.sample_contained_position()

  def state(self):
    global_state = {
        'success': self.success(),
    }
    if self._metadata:
      global_state['metadata'] = self._metadata
    return {'sprites': self._sprites, 'global_state': global_state}

  def observation(self):
    state = self.state()
    observation = {
        name: renderer.render(**state)
        for name, renderer in six.iteritems(self._renderers)
    }
    return observation

  def observation_spec(self):
    if not self._renderers_initialized:
      # Force a rendering so that the sizes of observeration_specs are correct.
      self.observation()
      self._renderers_initialized = True

    renderer_spec = {
        name: renderer.observation_spec()
        for name, renderer in six.iteritems(self._renderers)
    }
    return renderer_spec

  def action_spec(self):
    return self._action_space.action_spec()

  @property
  def action_space(self):
    return self._action_space
