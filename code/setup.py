#!/usr/bin/env python

from setuptools import setup
import xmap

setup(
    name='xmap',
    version=str(xmap.__version__),
    description='xmap',
    author='Tao LIN',
    author_email='itamtao@gmail.com',
    url='https://github.com/freeman-lab/spark-ml-streaming',
    packages=['xmap', 'xmap.core', 'xmap.utils'],
    scripts=[],
    keywords=['xmap'],
    package_data={},
    install_requires=open('requirements.txt').read().split()
)
