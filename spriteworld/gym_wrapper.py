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
"""Wrapper to make Spriteworld conform to the OpenAI Gym interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
from gym import spaces
import numpy as np


def _spec_to_space(spec):
  """Convert dm_env.specs to gym.Spaces."""
  if isinstance(spec, list):
    return spaces.Tuple([_spec_to_space(s) for s in spec])
  elif isinstance(spec, specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, specs.BoundedArray):
    return spaces.Box(
        np.asscalar(spec.minimum),
        np.asscalar(spec.maximum),
        shape=spec.shape,
        dtype=spec.dtype)
  else:
    raise ValueError('Unknown type for specs: {}'.format(spec))


class GymWrapper(object):
  """Wraps a Spriteworld environment into a Gym interface.

  Observations will be a dictionary, with the same keys as the 'renderers' dict
  provided when constructing a Spriteworld environment. Rendering is always
  performed, so calling render() is a no-op.
  """
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self, env):
    self._env = env
    self._last_render = None
    self._action_space = None
    self._observation_space = None

    # Reset Spriteworld to setup the observation_specs correctly
    self._env.reset()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    if self._observation_space is None:
      components = {}
      for key, value in self._env.observation_spec().items():
        components[key] = spaces.Box(
            -np.inf, np.inf, value.shape, dtype=value.dtype)
      self._observation_space = spaces.Dict(components)
    return self._observation_space

  @property
  def action_space(self):
    if self._action_space is None:
      self._action_space = _spec_to_space(self._env.action_spec())
    return self._action_space

  def _process_obs(self, obs):
    """Convert and processes observations."""
    for k, v in obs.items():
      obs[k] = np.asarray(v)
      if obs[k].dtype == np.bool:
        # Convert boolean 'success' into an float32 to predict it.
        obs[k] = obs[k].astype(np.float32)
      if k == 'image':
        self._last_render = obs[k]

    return obs

  def step(self, action):
    """Main step function for the environment.

    Args:
      action: Array R^4

    Returns:
      obs: dict of observations. Follows from the 'renderers' configuration
        provided as parameters to Spriteworld.
      reward: scalar reward.
      done: True if terminal state.
      info: dict with extra information (e.g. discount factor).
    """
    time_step = self._env.step(action)
    obs = self._process_obs(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    """Reset environment.

    Returns:
      obs: dict of observations. Follows from the 'renderers' configuration
        provided as parameters to Spriteworld.
    """
    time_step = self._env.reset()
    return self._process_obs(time_step.observation)

  def render(self, mode='rgb_array'):
    """Render function, noop for compatibility.

    Args:
      mode: unused, always returns an RGB array.

    Returns:
      Last RGB observation (cached from last observation with key 'image')
    """
    del mode
    return self._last_render

  def close(self):
    """Unused."""
    pass
