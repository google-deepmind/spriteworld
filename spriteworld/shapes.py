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
"""Shapes for Spriteworld.

Contains functions that generate np.arrays containing vertex arrays for various
sprite shapes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range


def _polar2cartesian(r, theta):
  return r * np.array([np.cos(theta), np.sin(theta)])


def polygon(num_sides, theta_0=0.):
  """Generate the vertices of a regular polygon.

  Args:
    num_sides: Int. Number of sides of the polygon.
    theta_0: Float. Initial angle to start the vertices from.

  Returns:
    path: Array of vertices of the polygon, normalized so it has area 1.
  """
  theta = 2 * np.pi / num_sides
  path = np.array(
      [_polar2cartesian(1, i * theta + theta_0) for i in range(num_sides)])
  area = num_sides * np.sin(theta / 2) * np.cos(theta / 2)
  path = np.array(path) / np.sqrt(area)
  return path


def star(num_sides, point_height=1, theta_0=0.):
  """Generate the vertices of a regular star shape.

  Args:
    num_sides: Int. Number of sides (i.e. number of points) in the star.
    point_height: Scalar. Height of each point of the star, relative to the
      radius of the star's inscribed circle.
    theta_0: Float. Initial angle to start the vertices from.

  Returns:
    path: Array of vertices of the star, normalized so the star has area 1.
  """
  point_to_center = 1 + point_height
  theta = 2 * np.pi / num_sides
  path = np.empty([2 * num_sides, 2])
  for i in range(num_sides):
    path[2 * i] = _polar2cartesian(1, i * theta + theta_0)
    path[2 * i + 1] = _polar2cartesian(point_to_center,
                                       (i + 0.5) * theta + theta_0)

  area = point_to_center * num_sides * np.sin(theta / 2)
  path = np.array(path) / np.sqrt(area)
  return path


def spokes(num_sides, spoke_height=1, theta_0=0.):
  """Generate the vertices of a regular rectangular spoke shape.

  This is like a star, except the points are rectangular. For example, if
  num_sides = 4, it will look like this:

                            O       O
                          O   O   O   O
                        O       O       O
                          O           O
                            O       O
                          O           O
                        O       O       O
                          O   O   O   O
                            O       O

  Args:
    num_sides: Int. Number of sides (i.e. number of points) in the star.
    spoke_height: Scalar. Height of each spoke, relative to the radius of the
      spoke shape's inscribed circle.
    theta_0: Float. Initial angle to start the vertices from.

  Returns:
    path: Array of vertices of the spoke shape, normalized so the spoke shape
      has area 1.
  """
  theta = 2 * np.pi / num_sides
  path = np.empty([3 * num_sides, 2])
  spoke = _polar2cartesian(spoke_height, -0.5 * theta + theta_0)
  for i in range(num_sides):
    vertex = _polar2cartesian(1, i * theta + theta_0)
    path[3 * i] = spoke + vertex
    path[3 * i + 1] = vertex
    spoke = _polar2cartesian(spoke_height, (i + 0.5) * theta + theta_0)
    path[3 * i + 2] = spoke + vertex

  area = num_sides * np.sin(theta / 2) * (2 + np.cos(theta / 2))

  path = np.array(path) / np.sqrt(area)
  return path
