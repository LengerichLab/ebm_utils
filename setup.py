"""
Intalling EBM Utils.
"""

from setuptools import setup, find_packages

setup(name='ebm_utils',
      packages=find_packages(),
      version='0.0.1',
      author='Ben Lengerich',
      url='github.com/blengerich/ebm_utils',
      install_requires=[
          'numpy',
          'scikit-learn',
          'matplotlib',
          'pandas',
          'interpret>=0.2.0',
          'numpy>=1.19.2',
          'ipywidgets',
          'scanpy',
      ],
)
