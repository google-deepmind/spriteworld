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
"""Installation script for setuptools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

setup(
    name='spriteworld',
    version='1.0.1',
    description=('Spriteworld is a python-based reinforcement learning '
                 'environment consisting of a 2-dimensional arena with objects '
                 'that can be freely moved.'),
    author='DeepMind',
    url='https://github.com/deepmind/spriteworld/',
    license='Apache License, Version 2.0',
    keywords=[
        'ai',
        'reinforcement-learning',
        'python',
        'machine learning',
        'objects',
    ],
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    install_requires=[
        'absl-py',
        'dm_env',
        'enum34',
        'matplotlib',
        'mock',
        'numpy',
        'pillow',
        'scikit-learn',
        'six',
    ],
    tests_require=[
        'nose',
        'absl-py',
    ],
    extras_require={
        'gym': ['gym'],
    },
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
