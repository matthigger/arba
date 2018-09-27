from setuptools import setup

setup(name='pnl_segment',
      version='0.1',
      description='pnl segmentation tool',
      url='https://github.com/matthigger/pnl_segment',
      author='matt higger',
      author_email='matt.higger@gmail.com',
      license='MIT',
      packages=['pnl_segment'],
      install_requires=[
          'numpy', 
          'sortedcontainers',
          'tqdm',
          'nibabel',
          'networkx',
          'scipy',
          'matplotlib',
          'sklearn'],
      zip_safe=False)
