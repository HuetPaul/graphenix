import matplotlib
import numpy
import scipy
from setuptools import find_packages, setup

setup(
    name='graphix',
    packages=find_packages(include=['graphix']),
    version='0.2.0',
    description='A library to use matplotlib more easily',
    author='Paul Huet',
    install_requires=[numpy, matplotlib, scipy],
    setup_requires=['pytest-runner'],
)
