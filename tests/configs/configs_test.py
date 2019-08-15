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
"""Tests for task configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from six.moves import range
from spriteworld import environment
from spriteworld.configs import cobra
from spriteworld.configs import examples


class ConfigsTest(parameterized.TestCase):

  @parameterized.parameters(
      (cobra.exploration,),
      (cobra.goal_finding_more_distractors,),
      (cobra.goal_finding_more_targets,),
      (cobra.goal_finding_new_position,),
      (cobra.goal_finding_new_shape,),
      (cobra.clustering,),
      (cobra.sorting,),
      (examples.goal_finding_embodied,),
  )
  def testConfig(self, task_module, modes=('train', 'test'), replicas=3):
    for mode in modes:
      print(mode)
      for _ in range(replicas):
        config = task_module.get_config(mode=mode)
        config['renderers'] = {}
        env = environment.Environment(**config)
        env.observation_spec()
        action = env.action_space.sample()

        num_episodes = 0
        step = env.reset()
        while num_episodes < 5:
          if step.first():
            num_episodes += 1
          step = env.step(action)


if __name__ == '__main__':
  absltest.main()
