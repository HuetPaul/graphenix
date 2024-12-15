
from setuptools import find_packages, setup

setup(
    name='illuspy',
    packages=find_packages(include=['illuspy']),
    version='0.2.0',
    description='A library to use matplotlib more easily',
    author='Paul Huet',
    setup_requires=['pytest-runner'],
)
