import setuptools

setuptools.setup(name='ebm_utils',
      packages=['ebm_utils'],
      version='0.0.0',
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
