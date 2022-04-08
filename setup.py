import setuptools

setuptools.setup(name='ebm_utils',
      packages=['ebm_utils'],
      version='0.0.0',
      install_requires=[
          'numpy',
          'tqdm',
          'scikit-learn',
          'python-igraph',
          'matplotlib',
          'pandas',
          'umap-learn',
          'interpret',
          'tensorflow>=2.4.0',
          'tensorflow-addons',
          'numpy>=1.19.2',
          'ipywidgets',
          'torchvision',
          'scanpy',
      ],
)
