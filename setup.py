from setuptools import setup
from setuptools import find_packages


setup(name='deeplearning4ir',
      version='0',
      description='deep learning for IR',
      author='Chenyan Xiong',
      install_requires=['scikit-learn', 'sklearn', 'numpy', 'scipy', 'traitlets', 'nltk', 'keras',
                        'gensim', 'pynlpir'
                        ],
      packages=find_packages()
      )