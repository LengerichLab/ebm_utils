"""
Intalling EBM Utils.
"""

from setuptools import setup, find_packages

VERSION = "0.0.1"

setup(
    name='ebm_utils',
    packages=find_packages(),
    version=VERSION,
    author='Ben Lengerich',
    url='github.com/blengerich/ebm_utils',
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'pandas',
        'interpret>=0.2.0',
        'numpy>=1.19.2',
        'ipywidgets',
        'scanpy',
        'ruptures',
      ],
)
