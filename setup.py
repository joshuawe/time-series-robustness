from setuptools import setup, find_packages

setup(
    name='tsrobustness',
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*'])
)