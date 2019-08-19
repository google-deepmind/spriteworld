# pylint: disable=g-bad-file-header
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
"""Start demo GUI for Spriteworld task configs.

To play a task, run this on the task config:
```bash
python run_demo.py --config=$path_to_task_config$
```

Be aware that this demo overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from spriteworld import demo_ui

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'spriteworld.configs.cobra.clustering',
                    'Module name of task config to use.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')
flags.DEFINE_boolean('task_hsv_colors', True,
                     'Whether the task config uses HSV as color factors.')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 10, 'Renderer anti-aliasing factor.')


def main(_):
  config = importlib.import_module(FLAGS.config)
  config = config.get_config(FLAGS.mode)
  demo_ui.setup_run_ui(config, FLAGS.render_size, FLAGS.task_hsv_colors,
                       FLAGS.anti_aliasing)


if __name__ == '__main__':
  app.run(main)
