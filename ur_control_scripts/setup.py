#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['camera', 'ur_control_moveit', 'motion_capture', 'utils', 'learning'],
    package_dir={'': 'src'}
)

setup(**d)
