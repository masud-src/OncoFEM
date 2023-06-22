from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='oncofem',
    version='0.1',
    author='Marlon Suditsch',
    author_email='marlon.suditsch@mechbau.uni-stuttgart.de',
    description='OncoFEM is a software package for patient-specific simulation',
    packages=find_packages(),
    install_requires=requirements,
)
