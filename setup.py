# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE

from setuptools import setup
from setuptools import find_packages


setup(name='knrm',
      version='0',
      description='knrm',
      author='Zhuyun Dai and Chenyan Xiong',
      install_requires=['numpy', 'traitlets', 'tensorflow'],
      packages=find_packages()
      )
