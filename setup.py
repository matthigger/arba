from setuptools import setup

setup(name='arba',
      version='0.1',
      description='pnl segmentation tool',
      url='https://github.com/matthigger/arba',
      author='matt higger',
      author_email='matt.higger@gmail.com',
      license='MIT',
      packages=['arba'],
      install_requires=[
          'numpy',
          'sortedcontainers',
          'tqdm',
          'pytest',
          'statsmodels',
          'nibabel',
          'networkx',
          'scipy',
          'matplotlib',
          'sklearn'],
      zip_safe=False)
