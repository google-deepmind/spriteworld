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
"""Abstract base class for renderers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractRenderer(object):
  """Abstract base class for renderers."""

  @abc.abstractmethod
  def render(self, sprites=(), global_state=None):
    """Renderer the sprites and global_state.

    Args:
      sprites: Iterable of sprites to be rendered.
      global_state: May contain extra information for rendering (e.g.
        background, symbolic/linguistic data, etc.).
    """

  @abc.abstractmethod
  def observation_spec(self):
    """Get observation spec for the output.

    Returns:
      ArraySpec or nested structure of such. Must agree with the output of
        self.update().
    """
