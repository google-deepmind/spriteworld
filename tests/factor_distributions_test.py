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
"""Tests for factor_distributions.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from six.moves import range
from spriteworld import factor_distributions as distribs


def test_sampling_and_containment(test_object, d, contained, not_contained):
  for _ in range(5):
    test_object.assertTrue(d.contains(d.sample()))
  for contained_spec in contained:
    try:
      contained = d.contains(contained_spec)
    except KeyError:
      # Having the wrong keys also indicate it is not contained.
      contained = False
    test_object.assertTrue(contained)
  for not_contained_spec in not_contained:
    try:
      contained = d.contains(not_contained_spec)
    except KeyError:
      contained = False
    test_object.assertFalse(contained)


class ContinuousTest(parameterized.TestCase):
  """Runs tests for Continuous distribution."""

  @parameterized.parameters(
      (0.0, 1.0, (-0.5, 2.0)),
      (-1.0, -0.1, (-1.4, 0.5)),
  )
  def testSamplingContainmentContinuous(self, minval, maxval, not_contained):
    d = distribs.Continuous('x', minval, maxval)
    for _ in range(5):
      self.assertTrue(d.contains(d.sample()))
    for not_contained_value in not_contained:
      self.assertFalse(d.contains({'x': not_contained_value}))

  def testDType(self):
    d_int = distribs.Continuous('x', 0, 1, dtype='int32')
    d_float = distribs.Continuous('x', 0, 1, dtype='float32')
    self.assertTrue(d_int.contains({'x': 0}))
    self.assertTrue(d_float.contains({'x': 0}))
    self.assertFalse(d_int.contains({'x': 1}))
    self.assertFalse(d_float.contains({'x': 1}))
    for _ in range(5):
      self.assertEqual(d_int.sample(), {'x': 0})


class DiscreteTest(parameterized.TestCase):
  """Runs tests for Discrete distribution."""

  @parameterized.parameters(
      ([1, 2, 3], [1, 2, 3], [0, 4]),
      (['a', 'b', 'c'], ['a', 'b', 'c'], ['d', 0]),
  )
  def testSamplingContainmentDiscrete(self, candidates, contained,
                                      not_contained):
    d = distribs.Discrete('x', candidates)
    cont = [{'x': value} for value in contained]
    not_cont = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, cont, not_cont)


class MixtureTest(parameterized.TestCase):
  """Runs tests for Mixture of distributions."""

  @parameterized.named_parameters(
      ('DisjointContinuous', distribs.Continuous(
          'x', 0, 1), distribs.Continuous('x', 2, 3), [0.5, 2.5], [1.5, 3.5]),
      ('OverlappingContinuous', distribs.Continuous(
          'x', 0, 2), distribs.Continuous('x', 1, 3), [0.5, 2.5], [-0.5, 3.5]),
      ('DisjointDiscrete', distribs.Discrete(
          'x', [0, 1]), distribs.Discrete('x', [2, 3]), [1, 2], [-1, 4]),
      ('OverlappingDiscrete', distribs.Discrete(
          'x', [0, 1]), distribs.Discrete('x', [1, 2]), [0, 1, 2], [-1, 3]),
      ('ContinuousDiscrete', distribs.Continuous(
          'x', 0, 2), distribs.Discrete('x', [1, 3]), [0.5, 3], [2.5]),
  )
  def testSamplingContainmentMixtureTwo(self, c_0, c_1, contained,
                                        not_contained):
    d = distribs.Mixture((c_0, c_1))
    contained = [{'x': value} for value in contained]
    not_contained = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, contained, not_contained)

  def testSamplingContainmentMixtureMultiple(self):
    dists = [
        distribs.Continuous('x', 0, 2),
        distribs.Continuous('x', 1, 5),
        distribs.Continuous('x', 9, 12),
        distribs.Discrete('x', [7, 10]),
        distribs.Discrete('x', [14]),
    ]
    contained = [0.5, 4, 11, 7, 14]
    not_contained = [5.5, 6, 8, 13]
    d = distribs.Mixture(dists)
    contained = [{'x': value} for value in contained]
    not_contained = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, contained, not_contained)

  def testRaisesError(self):
    c_0 = distribs.Discrete('x', [0])
    c_1 = distribs.Discrete('y', [1])
    with self.assertRaises(ValueError):
      distribs.Mixture((c_0, c_1))

  def testProbs(self):
    c_0 = distribs.Discrete('x', [0])
    c_1 = distribs.Discrete('x', [1])
    d_0 = distribs.Mixture([c_0, c_1], probs=(0.3, 0.7))
    d_1 = distribs.Mixture([c_0, c_1], probs=(0.0, 1.0))
    for _ in range(5):
      self.assertTrue(d_0.contains(d_0.sample()))
    for _ in range(5):
      self.assertEqual(d_1.sample(), {'x': 1})


class IntersectionTest(parameterized.TestCase):
  """Runs tests for Intersection of distributions."""

  @parameterized.named_parameters(
      ('ContinuousContinuous', distribs.Continuous(
          'x', 0, 2), distribs.Continuous('x', 1, 3), [1.5], [0.5, 2.5]),
      ('DiscreteDiscrete', distribs.Discrete(
          'x', [0, 1]), distribs.Discrete('x', [1, 2]), [1], [0, 2]),
      ('DiscreteContinuous', distribs.Discrete(
          'x', [1, 3]), distribs.Continuous('x', 0, 2), [1], [0.5, 1.5, 3]),
  )
  def testSamplingContainmentIntersectionTwo(self, d_0, d_1, contained,
                                             not_contained):
    d = distribs.Intersection((d_0, d_1))
    contained = [{'x': value} for value in contained]
    not_contained = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, contained, not_contained)

  def testSamplingContainmentIntersectionMultiple(self):
    dists = [
        distribs.Discrete('x', [1, 2.5, 3, 4, 6]),
        distribs.Discrete('x', [1, 2.5, 3, 12]),
        distribs.Continuous('x', 0, 5),
        distribs.Continuous('x', 2, 10),
    ]
    contained = [2.5, 3]
    not_contained = [1, 4, 8]
    d = distribs.Intersection(dists)
    contained = [{'x': value} for value in contained]
    not_contained = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, contained, not_contained)

  def testRaisesError(self):
    d_0 = distribs.Continuous('x', 0, 1)
    d_1 = distribs.Continuous('x', 2, 3)
    d = distribs.Intersection((d_0, d_1))
    with self.assertRaises(ValueError):
      d.sample()

  def testIndexForSampling(self):
    d_0 = distribs.Continuous('x', 0, 2)
    d_1 = distribs.Discrete('x', [1, 3])
    d = distribs.Intersection((d_0, d_1), index_for_sampling=1)
    d.sample()
    with self.assertRaises(ValueError):
      d = distribs.Intersection((d_0, d_1), index_for_sampling=0)
      d.sample()

  def testKeys(self):
    d_0 = distribs.Product(
        (distribs.Continuous('x', 0, 2), distribs.Continuous('y', 0, 1)))
    d_1 = distribs.Product(
        (distribs.Continuous('x', 0, 1), distribs.Continuous('y', 0, 0.5)))
    d_2 = distribs.Continuous('x', 0.4, 0.6)

    distribs.Intersection((d_0, d_1))
    with self.assertRaises(ValueError):
      distribs.Intersection((d_0, d_2))


class SelectionTest(parameterized.TestCase):
  """Runs tests for Selection of distributions."""

  @parameterized.named_parameters(
      (
          'Continuous',
          distribs.Continuous('x', 0, 2),
          distribs.Continuous('x', 1, 3),
          [{
              'x': 1.5
          }],
          [{
              'x': 0.5
          }, {
              'x': 2.5
          }],
      ),
      (
          'Discrete',
          distribs.Discrete('x', [0, 1]),
          distribs.Discrete('x', [1, 2]),
          [{
              'x': 1
          }],
          [{
              'x': 0
          }, {
              'x': 2
          }],
      ),
      (
          'MultiDimensional',
          distribs.Product(
              (distribs.Discrete('x', [1, 2]), distribs.Discrete('y', [3, 4]))),
          distribs.Discrete('x', [2]),
          [{
              'x': 2,
              'y': 3
          }],
          [{
              'x': 1,
              'y': 3
          }, {
              'x': 2
          }, {
              'x': 2,
              'y': 5
          }],
      ),
  )
  def testSamplingContainmentSelection(self, d_base, d_filter, contained,
                                       not_contained):
    d = distribs.Selection(d_base, d_filter)
    test_sampling_and_containment(self, d, contained, not_contained)

  def testRaisesErrorFailedSampling(self):
    d_base = distribs.Continuous('x', 0, 1)
    d_filter = distribs.Continuous('x', 2, 3)
    d = distribs.Selection(d_base, d_filter)
    with self.assertRaises(ValueError):
      d.sample()

  def testKeys(self):
    d_base = distribs.Product(
        (distribs.Continuous('x', 0, 2), distribs.Continuous('y', 0, 1)))
    d_filter_1 = distribs.Continuous('x', 0, 1)
    d_filter_2 = distribs.Continuous('z', 0.4, 0.6)
    distribs.Selection(d_base, d_filter_1)
    with self.assertRaises(ValueError):
      distribs.Selection(d_base, d_filter_2)


class ProductTest(parameterized.TestCase):
  """Runs tests for Product of distributions."""

  @parameterized.named_parameters(
      ('ContinuousContinuous', distribs.Continuous(
          'x', 0, 2), distribs.Continuous('y', 1, 3), [{
              'x': 0.5,
              'y': 2.5
          }, {
              'x': 1.5,
              'y': 1.5
          }], [{
              'x': 0.5,
              'y': 0.5
          }, {
              'x': 2.5,
              'y': 1.5
          }]),
      ('DiscreteDiscrete', distribs.Discrete(
          'x', [0, 1]), distribs.Discrete('y', [1, 2]), [{
              'x': 0,
              'y': 2
          }, {
              'x': 1,
              'y': 1
          }], [{
              'x': 1,
              'y': 0
          }, {
              'x': 2,
              'y': 2
          }]),
      ('DiscreteContinuous', distribs.Discrete(
          'x', [1, 3]), distribs.Continuous('y', 0, 2), [{
              'x': 1,
              'y': 1
          }, {
              'x': 3,
              'y': 0.5
          }], [{
              'x': 2,
              'y': 1
          }, {
              'x': 3,
              'y': 3
          }]),
  )
  def testSamplingContainmentProductTwo(self, d_0, d_1, contained,
                                        not_contained):
    d = distribs.Product((d_0, d_1))
    test_sampling_and_containment(self, d, contained, not_contained)

  def testSamplingContainmentProductMultiple(self):
    dists = [
        distribs.Discrete('x', [1, 2.5, 3, 4, 6]),
        distribs.Discrete('y', [1, 2.5, 3, 12]),
        distribs.Continuous('z', 0, 5),
        distribs.Continuous('w', 2, 10),
    ]
    contained = [{'x': 2.5, 'y': 12, 'z': 3.5, 'w': 9}]
    not_contained = [
        {'x': 2.5, 'y': 12, 'z': 3.5, 'w': 1},
        {'x': 3.5, 'y': 12, 'z': 3.5, 'w': 9},
    ]
    d = distribs.Product(dists)
    test_sampling_and_containment(self, d, contained, not_contained)

  def testRaisesError(self):
    d_0 = distribs.Continuous('x', 0, 1)
    d_1 = distribs.Continuous('x', 2, 3)
    with self.assertRaises(ValueError):
      distribs.Product((d_0, d_1))

  def testkeys(self):
    dists = [
        distribs.Discrete('x', [1, 2.5, 3, 12]),
        distribs.Continuous('y', 0, 5),
        distribs.Continuous('z', 2, 10),
    ]
    d = distribs.Product(dists)
    self.assertEqual(d.keys, set(('x', 'y', 'z')))


class SetMinusTest(parameterized.TestCase):
  """Runs tests for SetMinus of distributions."""

  @parameterized.named_parameters(
      ('ContinuousContinuous', distribs.Continuous(
          'x', 0, 2), distribs.Continuous('x', 1, 3), [0.5], [1.5]),
      ('DiscreteDiscrete', distribs.Discrete(
          'x', [0, 1]), distribs.Discrete('x', [1, 2]), [0], [1]),
      ('DiscreteContinuous', distribs.Discrete(
          'x', [1, 3]), distribs.Continuous('x', 0, 2), [3], [1]),
      ('ContinuousDiscrete', distribs.Continuous(
          'x', 0, 2), distribs.Discrete('x', [1, 3]), [0.5, 1.5], [1]),
  )
  def testSamplingContainmentSetMinusTwo(self, d_0, d_1, contained,
                                         not_contained):
    d = distribs.SetMinus(d_0, d_1)
    contained = [{'x': value} for value in contained]
    not_contained = [{'x': value} for value in not_contained]
    test_sampling_and_containment(self, d, contained, not_contained)

  def testSamplingContainmentSetMinusMultiple(self):
    base = distribs.Continuous('x', 2, 10)
    hold_out = distribs.Mixture([
        distribs.Discrete('x', [1, 4, 6]),
        distribs.Discrete('x', [3, 8, 9, 12]),
        distribs.Continuous('x', 3, 5),
    ])
    contained = [{'x': value} for value in [2.5, 5.5, 7, 9.5]]
    not_contained = [{'x': value} for value in [4, 6, 9, 11]]
    d = distribs.SetMinus(base, hold_out)
    test_sampling_and_containment(self, d, contained, not_contained)

  def testRaisesError(self):
    d_0 = distribs.Continuous('x', 0, 2)
    d_1 = distribs.Continuous('y', 1, 3)
    with self.assertRaises(ValueError):
      distribs.SetMinus(d_0, d_1)


class CompositionTest(parameterized.TestCase):
  """Runs tests for compositions of distribution operations."""

  def testCornerUnion(self):
    square_0 = distribs.Product([
        distribs.Continuous('x', 0, 3),
        distribs.Continuous('y', 0, 3),
    ])
    hold_out_0 = distribs.Product([
        distribs.Continuous('x', 1, 3),
        distribs.Continuous('y', 0, 2),
    ])
    square_1 = distribs.Product([
        distribs.Continuous('x', 2, 5),
        distribs.Continuous('y', 0, 3),
    ])
    hold_out_1 = distribs.Product([
        distribs.Continuous('x', 2, 4),
        distribs.Continuous('y', 1, 3),
    ])
    corner_0 = distribs.SetMinus(square_0, hold_out_0)
    corner_1 = distribs.SetMinus(square_1, hold_out_1)
    corner_union = distribs.Mixture([corner_0, corner_1])

    contained = [
        {'x': 0.5, 'y': 0.5},
        {'x': 0.5, 'y': 2.5},
        {'x': 2.5, 'y': 2.5},
        {'x': 2.5, 'y': 0.5},
        {'x': 4.5, 'y': 0.5},
        {'x': 4.5, 'y': 2.5},
    ]
    not_contained = [
        {'x': 1.5, 'y': 0.5},
        {'x': 1.5, 'y': 1.5},
        {'x': 2.5, 'y': 1.5},
        {'x': 3.5, 'y': 1.5},
        {'x': 3.5, 'y': 2.5},
    ]
    test_sampling_and_containment(self, corner_union, contained, not_contained)

  def testCubeWithTunnel(self):
    cube = distribs.Product([
        distribs.Continuous('x', 0, 1),
        distribs.Continuous('y', 0, 1),
        distribs.Continuous('z', 0, 1),
    ])
    tunnel = distribs.Product([
        distribs.Continuous('x', 0.25, 0.75),
        distribs.Continuous('y', 0.25, 0.75),
    ])
    cube_with_tunnel = distribs.SetMinus(cube, tunnel)

    contained = [
        {'x': 0.2, 'y': 0.2, 'z': 0.2},
        {'x': 0.2, 'y': 0.2, 'z': 0.5},
        {'x': 0.2, 'y': 0.5, 'z': 0.5}
    ]
    not_contained = [
        {'x': 0.5, 'y': 0.5, 'z': 0.5},
        {'x': 0.5, 'y': 0.5, 'z': 0.2},
    ]
    test_sampling_and_containment(self, cube_with_tunnel, contained,
                                  not_contained)


if __name__ == '__main__':
  absltest.main()
