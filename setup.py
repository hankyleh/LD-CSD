from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='LDCSD',
    version='0.0.0',
    description='CSD Transport Solver with LD in energy',
    long_description=long_description,
    license='BSD-3',
    author='Kyle Hansen',
    author_email='khansen3@ncsu.edu',
    packages=['LDCSD'],
    install_requires=['matplotlib','numpy']
)
