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
"""Factor distribution library.

This library contains classes for defining distributions of sprite factors.
A number of set-theoretic operations are supported, with which it is possible to
define factor distributions that are arbitrarily nested mixtures, intersections,
products, and differences of single-factor continuous/discrete distributions.

A factor specification is called a "spec", which is a dictionary of sprite
factors, hence can have keys such as "size", "shape", "x_pos", etc. However, the
classes in this file are general and make no reference to the particular factor
names used by Spriteworld sprites.

All distributions inherit from AbstractDistribution. They have a "sample()"
method, which returns a spec. The keys of this spec can be accessed by the
"keys" property. Distributions also have a "contains(spec)" method, which checks
if the argument "spec" is in the support of the distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import numpy as np
import six

# Maximum number of tries used for rejection sampling from Intersection and
# SetMinus distributions
_MAX_TRIES = int(1e5)


@six.add_metaclass(abc.ABCMeta)
class AbstractDistribution(object):
  """Abstract class from which all distributions should inherit."""

  @abc.abstractmethod
  def sample(self, rng=None):
    """Sample a spec from this distribution. Returns a dictionary.

    Args:
      rng: Random number generator. Fed into self._get_rng(), if None defaults
        to np.random.
    """

  @abc.abstractmethod
  def contains(self, spec):
    """Return whether distribution contains spec dictionary."""

  @abc.abstractmethod
  def to_str(self, indent):
    """Recursive string description of this distribution."""

  def __str__(self):
    return self.to_str(indent=0)

  def _get_rng(self, rng=None):
    """Get random number generator, defaulting to np.random."""
    return np.random if rng is None else rng

  @abc.abstractproperty
  def keys(self):
    """The set of keys in specs sampled from this distribution."""


class Continuous(AbstractDistribution):
  """Continuous 1-dimensional uniform distribution."""

  def __init__(self, key, minval, maxval, dtype='float32'):
    """Construct continuous 1-dimensional uniform distribution.

    Args:
      key: String factor name. self.sample() returns {key: _}.
      minval: Scalar minimum value.
      maxval: Scalar maximum value.
      dtype: String numpy dtype.
    """
    self.key = key
    self.minval = minval
    self.maxval = maxval
    self.dtype = dtype

  def sample(self, rng=None):
    """Sample value in [self.minval, self.maxval) and return dict."""
    rng = self._get_rng(rng)
    out = rng.uniform(low=self.minval, high=self.maxval)
    out = np.cast[self.dtype](out)
    return {self.key: out}

  def contains(self, spec):
    """Check if spec[self.key] is in [self.minval, self.maxval)."""
    if self.key not in spec:
      raise ValueError('key {} is not in spec {}, but must be to evaluate '
                       'containment.'.format(self.key, spec))
    else:
      return spec[self.key] >= self.minval and spec[self.key] < self.maxval

  def to_str(self, indent):
    s = '<Continuous: key={}, mival={}, maxval={}, dtype={}>'.format(
        self.key, self.minval, self.maxval, self.dtype)
    return indent * '  ' + s

  @property
  def keys(self):
    return set([self.key])


class Discrete(AbstractDistribution):
  """Discrete distribution."""

  def __init__(self, key, candidates, probs=None):
    """Construct discrete distribution.

    Args:
      key: String. Factor name.
      candidates: Iterable. Discrete values to sample from.
      probs: None or iterable of floats summing to 1. Candidate sampling
        probabilities. If None, candidates are sampled uniformly.
    """
    self.candidates = candidates
    self.key = key
    self.probs = probs

  def sample(self, rng=None):
    rng = self._get_rng(rng)
    out = self.candidates[rng.choice(len(self.candidates), p=self.probs)]
    return {self.key: out}

  def contains(self, spec):
    if self.key not in spec:
      raise KeyError('key {} is not in spec {}, but must be to evaluate '
                     'containment.'.format(self.key, spec))
    else:
      return spec[self.key] in self.candidates

  def to_str(self, indent):
    s = '<Discrete: key={}, candidates={}, probs={}>'.format(
        self.key, self.candidates, self.probs)
    return indent * '  ' + s

  @property
  def keys(self):
    return set([self.key])


class Mixture(AbstractDistribution):
  """Mixture of distributions."""

  def __init__(self, components, probs=None):
    """Construct mixture of distributions.

    This is a mixture distribution, not a union, so if the components overlap,
    their overlap will be sampled more than the non-overlapping regions.

    Args:
      components: Iterable of component distributions. Must all have the same
        key sets.
      probs: None or iterable of floats summing to 1. Sampling probabilities for
        the components.
    """
    self.components = components
    if probs is None:
      self.probs = np.ones(len(components)) / len(components)
    else:
      self.probs = np.array(probs)

    self._keys = components[0].keys
    for c in components[1:]:
      if c.keys != self._keys:
        raise ValueError(
            'All components must have the same key sets. However detected key '
            'sets {} and {}'.format(self._keys, c.keys))

  def sample(self, rng=None):
    rng = self._get_rng(rng)
    sample_index = rng.choice(len(self.components), p=self.probs)
    sample = self.components[sample_index].sample(rng=rng)
    return sample

  def contains(self, spec):
    return any(c.contains(spec) for c in self.components)

  def to_str(self, indent):
    components_strings = [x.to_str(indent + 2) for x in self.components]
    s = (indent * '  ' + '<Mixture:\n' +
         (indent + 1) * '  ' + 'components=[\n{},\n' +
         (indent + 1) * '  ' + '],\n' +
         (indent + 1) * '  ' + 'probs={}>').format(
             ',\n'.join(components_strings), self.probs)
    return s

  @property
  def keys(self):
    return self._keys


class Intersection(AbstractDistribution):
  """Intersection of component distributions."""

  def __init__(self, components, index_for_sampling=0):
    """Construct intersection of component distributions.

    Samples are generated by sampling from one of the components and then doing
    rejection with the others, so if the component being sampled has some
    non-uniformity (e.g. a mixture with non-uniform probs), that non-uniformity
    will be inherited by the intersection.

    Args:
     components: Iterable of distributions.
     index_for_sampling: Int. Index of the component to use for sampling. All
       other components will be used to reject its samples. For efficiency, the
       user should ensure index_for_sampling corresponds to the smallest
       component distribution.
    """
    self.components = components
    self.index_for_sampling = index_for_sampling

    self._keys = components[0].keys
    for c in components[1:]:
      if c.keys != self._keys:
        raise ValueError(
            'All components must have the same key sets. However detected key '
            'sets {} and {}'.format(self._keys, c.keys))

  def sample(self, rng=None):
    rng = self._get_rng(rng)
    tries = 0
    while tries < _MAX_TRIES:
      tries += 1
      sample = self.components[self.index_for_sampling].sample(rng=rng)
      if all(c.contains(sample) for c in self.components):
        return sample
    raise ValueError('Maximum number of tried exceeded when trying to sample '
                     'from {}.'.format(str(self)))

  def contains(self, spec):
    return all(c.contains(spec) for c in self.components)

  def to_str(self, indent):
    components_strings = [x.to_str(indent + 2) for x in self.components]
    s = (indent * '  ' + '<Intersection:\n' +
         (indent + 1) * '  ' + 'components=[\n{},\n' +
         (indent + 1) * '  ' + '],\n' +
         (indent + 1) * '  ' + 'index_for_sampling={}>').format(
             ',\n'.join(components_strings), self.index_for_sampling)
    return s

  @property
  def keys(self):
    return self._keys


class Product(AbstractDistribution):
  """Product distribution."""

  def __init__(self, components):
    """Construct product distribution.

    This is used to create distributions over larger numbers of factors by
    taking the product of components. The components must have disjoint key
    sets.

    Args:
      components: Iterable of distributions.
    """
    self.components = components

    self._keys = functools.reduce(set.union, [set(c.keys) for c in components])
    num_keys = sum(len(c.keys) for c in components)
    if len(self._keys) < num_keys:
      raise ValueError(
          'All components must have different keys, yet there are {} '
          'overlapping keys.'.format(num_keys - len(self._keys)))

  def sample(self, rng=None):
    rng = self._get_rng(rng)
    sample = {}
    for c in self.components:
      sample.update(c.sample(rng=rng))
    return sample

  def contains(self, spec):
    return all(c.contains(spec) for c in self.components)

  def to_str(self, indent):
    components_strings = [x.to_str(indent + 2) for x in self.components]
    s = (indent * '  ' + '<Product:\n' +
         (indent + 1) * '  ' + 'components=[\n{},\n' +
         (indent + 1) * '  ' + ']>').format(
             ',\n'.join(components_strings))
    return s

  @property
  def keys(self):
    return self._keys


class SetMinus(AbstractDistribution):
  """Setminus of distributions."""

  def __init__(self, base, hold_out):
    """Construct setminus of distributions..

    This uses rejection sampling to take the difference of two distributions.

    Args:
      base: Distribution from which candidate samples are drawn.
      hold_out: Distribution used to reject samples from base.
    """
    self.base = base
    self.hold_out = hold_out

    self._keys = base.keys
    if not hold_out.keys.issubset(self._keys):
      raise ValueError(
          'Keys {} of hold_out is not a subset of keys {} of SetMinus base '
          'distribution.'
          .format(hold_out.keys, base.keys))

  def sample(self, rng=None):
    rng = self._get_rng(rng)
    tries = 0
    while tries < _MAX_TRIES:
      tries += 1
      sample = self.base.sample(rng=rng)
      if not self.hold_out.contains(sample):
        return sample
    raise ValueError('Maximum number of tried exceeded when trying to sample '
                     'from {}.'.format(str(self)))

  def contains(self, spec):
    return self.base.contains(spec) and not self.hold_out.contains(spec)

  def to_str(self, indent):
    s = (indent * '  ' + '<SetMinus:\n' +
         (indent + 1) * '  ' + 'base=\n{},\n' +
         (indent + 1) * '  ' + 'hold_out=\n{}>').format(
             self.base.to_str(indent + 2), self.hold_out.to_str(indent + 2))
    return s

  @property
  def keys(self):
    return self._keys
