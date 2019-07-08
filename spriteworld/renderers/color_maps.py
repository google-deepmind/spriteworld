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
"""Functions to transform between color spaces for rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys
import numpy as np


def hsv_to_rgb(c):
  """Convert HSV tuple to RGB tuple."""
  return tuple((255 * np.array(colorsys.hsv_to_rgb(*c))).astype(np.uint8))
