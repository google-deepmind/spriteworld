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
"""Interactive GUI for Spriteworld.

Be aware that this UI overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as log
import sys
from absl import logging
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np

from spriteworld import action_spaces
from spriteworld import environment
from spriteworld import renderers


class MatplotlibUI(object):
  """Class for visualising the environment based on Matplotlib."""

  def __init__(self):
    self.rewards = 10 * [np.nan]
    self.rewards_bounds = [-10, 10]
    self.last_success = None

    plt.ion()
    self._fig = plt.figure(
        figsize=(9, 12), num='Spriteworld', facecolor='white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    self._ax_image = plt.subplot(gs[0])
    self._ax_image.axis('off')

    self._ax_scalar = plt.subplot(gs[1])
    self._ax_scalar.spines['right'].set_visible(False)
    self._ax_scalar.spines['top'].set_visible(False)
    self._ax_scalar.xaxis.set_ticks_position('bottom')
    self._ax_scalar.yaxis.set_ticks_position('left')
    self._setup_callbacks()

  def _setup_callbacks(self):
    """Default callbacks for the UI."""

    # Pressing escape should stop the UI
    def _onkeypress(event):
      if not event.inaxes:
        return
      if event.key == 'escape':
        # Stop UI
        logging.info('Pressed escape, stopping UI.')
        plt.close(self._fig)
        sys.exit()

    self._fig.canvas.mpl_connect('key_release_event', _onkeypress)

    # Disable default keyboard shortcuts
    for key in ('keymap.fullscreen', 'keymap.home', 'keymap.back',
                'keymap.forward', 'keymap.pan', 'keymap.zoom', 'keymap.save',
                'keymap.quit', 'keymap.grid', 'keymap.yscale', 'keymap.xscale',
                'keymap.all_axes'):
      plt.rcParams[key] = ''

    # Disable logging of some matplotlib events
    log.getLogger('matplotlib').setLevel('WARNING')

  def _draw_observation(self, image, action):
    """Draw the latest observation."""
    self._ax_image.clear()
    self._ax_image.imshow(image, origin='lower', interpolation='none')
    self._ax_image.set_xticks([])
    self._ax_image.set_yticks([])
    if action is not None:
      size = image.shape[0]
      self._ax_image.annotate(
          '',
          xy=action[:2] * size,  # Start of arrow
          xytext=action[2:] * size,  # End of arrow
          arrowprops={
              'arrowstyle': '<|-',
              'color': 'red',
              'lw': 4,
          })

    # Indicate success
    linewidth = 1
    color = 'black'
    if np.isnan(self.rewards[-1]):
      linewidth = 8
      color = 'green' if self.last_success else 'red'

    for sp in self._ax_image.spines.values():
      sp.set_color(color)
      sp.set_linewidth(linewidth)

  def _draw_rewards(self):
    """Draw the past rewards plot."""
    self._ax_scalar.clear()
    self._ax_scalar.set_ylabel('Rewards')
    self._ax_scalar.set_xlabel('Timestep')
    xs = np.arange(-len(self.rewards), 0)
    self._ax_scalar.set_xticks(xs)
    self._ax_scalar.axhline(y=0.0, color='lightgrey', linestyle='--')
    self._ax_scalar.stem(xs, self.rewards, basefmt=' ')

    self._ax_scalar.set_xlim((xs[0] - 1.0, xs[-1] + 1.0))
    self._ax_scalar.set_ylim(
        (self.rewards_bounds[0] - 1.0, self.rewards_bounds[1] + 1.0))

  def register_callback(self, event_name, callback):
    """Register a callback for the given event."""
    self._fig.canvas.mpl_connect(event_name, callback)

  def update(self, timestep, action):
    """Update the visualisation with the latest timestep and action."""
    reward = timestep.reward
    if reward is None:
      reward = np.nan
    self.rewards = self.rewards[1:] + [reward]
    self.rewards_bounds[0] = np.nanmin(
        [np.nanmin(self.rewards), self.rewards_bounds[0]])
    self.rewards_bounds[1] = np.nanmax(
        [np.nanmax(self.rewards), self.rewards_bounds[1]])
    self._draw_observation(timestep.observation['image'], action)
    self._draw_rewards()
    plt.show(block=False)

    self.last_success = timestep.observation['success']


class HumanDragAndDropAgent(object):
  """Demo agent for mouse-clicking interface with DragAndDrop action space."""

  def __init__(self, action_space, timeout=600):
    self._action_space = action_space
    self._size = None
    self._click = None
    self._timeout = timeout

  def help(self):
    logging.info('Click to select an object, then click again to select where '
                 'to move it.')

  def callbacks(self):
    """Get the matplotlib callbacks required by the agent."""

    def _onclick(event):
      x, y = event.xdata, event.ydata
      self._click = (x, y)
      return

    return {'button_press_event': _onclick}

  def begin_episode(self):
    logging.info('Starting episode')

  def step(self, timestep):
    """Take a step."""
    if self._size is None:
      self._size = timestep.observation['image'].shape[0]

    def _get_click():
      """Get mouse click."""
      while True:
        x = plt.waitforbuttonpress(timeout=self._timeout)
        if x is None:
          logging.info('Timed out. You took longer than %d seconds to click.',
                       self._timeout)
          click = None
        elif x:
          logging.info('You pressed a key, but were supposed to click with the '
                       'mouse.')
          self.help()
          click = None
        else:
          click = self._click
        return click

    def _get_action():
      """Get action from user."""
      logging.info('Select sprite')
      click_position = _get_click()
      logging.info('Select target')
      click_motion = _get_click()
      try:
        action = (1. / self._size) * np.concatenate(
            (click_position, click_motion)).astype(np.float32)
        if any(np.isnan(action)):
          raise ValueError
        self._action_space.action_spec().validate(action)
        return action
      except (ValueError, TypeError):
        logging.info('Select a valid action')
        return _get_action()

    action = _get_action()
    return action


class HumanEmbodiedAgent(object):
  """Demo agent for keyboard interface with Embodied action space."""

  MOTION_KEY_TO_ACTION = {
      'up': 0,
      'left': 1,
      'down': 2,
      'right': 3,
      'w': 0,
      'a': 1,
      's': 2,
      'd': 3
  }

  def __init__(self, action_space, timeout=600):
    self._action_space = action_space
    self._key_press = None
    self._carry = False
    self._movement = None
    self._timeout = timeout

  def help(self):
    logging.info('Use WASD/arrow keys to move, hold Space to carry.')

  def callbacks(self):
    """Get the matplotlib callbacks required by the agent."""

    def _onkeypress(event):
      if event.key in HumanEmbodiedAgent.MOTION_KEY_TO_ACTION:
        self._movement = HumanEmbodiedAgent.MOTION_KEY_TO_ACTION[event.key]
      elif event.key == ' ':
        self._carry = True
      elif event.key == 'escape':
        plt.close('all')
        sys.exit()
      else:
        self.help()

    def _onkeyrelease(event):
      if event.key == ' ':
        self._carry = False
      elif event.key in HumanEmbodiedAgent.MOTION_KEY_TO_ACTION:
        self._movement = None

    return {'key_press_event': _onkeypress, 'key_release_event': _onkeyrelease}

  def begin_episode(self):
    logging.info('Starting episode')

  def step(self, timestep):
    """Take a step."""
    del timestep  # Unused

    def _wait_for_movement_key_press():
      """Get key press."""
      ready = False
      while not ready:
        x = plt.waitforbuttonpress(timeout=self._timeout)
        if x is None:
          logging.info('Timed out. You took longer than %d seconds to click.',
                       self._timeout)
        elif not x:
          logging.info('You clicked, but you are supposed to use the Keyboard.')
          self.help()
        elif self._movement is not None:
          ready = True

    def _get_action():
      """Get action from user."""
      _wait_for_movement_key_press()

      action = (int(self._carry), self._movement)
      for spec, a in zip(self._action_space.action_spec(), action):
        spec.validate(a)
      return action

    return _get_action()


def setup_run_ui(env_config, render_size, task_hsv_colors, anti_aliasing):
  """Start a Demo UI given an env_config."""
  if isinstance(env_config['action_space'], action_spaces.SelectMove):
    # DragAndDrop is a bit easier to demo than the SelectMove action space
    env_config['action_space'] = action_spaces.DragAndDrop(scale=0.5)
    agent = HumanDragAndDropAgent(env_config['action_space'])
  elif isinstance(env_config['action_space'], action_spaces.Embodied):
    agent = HumanEmbodiedAgent(env_config['action_space'])
  else:
    raise ValueError(
        'Demo is not configured to run with action space {}.'.format(
            env_config['action_space']))
  env_config['renderers'] = {
      'image':
          renderers.PILRenderer(
              image_size=(render_size, render_size),
              color_to_rgb=renderers.color_maps.hsv_to_rgb
              if task_hsv_colors else None,
              anti_aliasing=anti_aliasing),
      'success':
          renderers.Success()
  }
  env = environment.Environment(**env_config)
  demo = MatplotlibUI()

  for event_name, callback in agent.callbacks().items():
    demo.register_callback(event_name, callback)

  # Start RL loop
  timestep = env.reset()
  demo.update(timestep, action=None)

  while True:
    action = agent.step(timestep)
    timestep = env.step(action)
    if isinstance(env_config['action_space'], action_spaces.DragAndDrop):
      demo.update(timestep, action)
    else:
      demo.update(timestep, None)
