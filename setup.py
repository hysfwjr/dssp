""" manages your requirements files
"""
import sys
from setuptools import setup

setup(
    name='dssp',
    version='0.1.0',
    url='https://github.com/hysfwjr/dssp',
    license='',
    author='hysfwjr',
    author_email='hysfwjr@163.com',
    description=__doc__.strip('\n'),
    #packages=[],
    scripts=['src/dssp.py'],
    zip_safe=False,
    platforms='any',
    install_requires=['docopt'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ]
)
