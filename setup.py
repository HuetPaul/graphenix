from setuptools import find_packages, setup

setup(
    name='graphiques',
    packages=find_packages(include=['graphiques']),
    version='0.1.0',
    description='A library to use matplotlib more easly',
    author='Paul Huet',
    install_requires=[],
    setup_requires=['pytest-runner'],
)