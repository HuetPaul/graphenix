
from setuptools import find_packages, setup

setup(
    name='graphenix',
    packages=find_packages(include=['graphenix']),
    version='0.2.0',
    description='A library to store and replot matplotlib plot easly and reproducibly',
    author='Paul Huet',
    setup_requires=['pytest-runner'],
)
