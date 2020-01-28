from distutils.core import setup
from setuptools import find_packages

setup(name='symnet',
      version='0.1',
      description='SymNet - An accessible deep learning framework',
      author='Rahul Yedida',
      author_email='y.rahul@outlook.com',
      keywords=['keras', 'deep learning'],
      url='https://github.com/yrahul3910/symnet',
      packages=find_packages(),
      install_requires=[
          'keras==2.2.4',
          'tensorflow==1.15.2',
          'numpy==1.16.4',
          'matplotlib==3.1.0',
          'pandas==0.24.2',
          'imageio==2.5.0',
          'scikit-learn==0.21.2',
          'scipy==1.3.0'
      ]
)