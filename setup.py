#!/usr/bin/env python3
import os
from setuptools import setup, find_packages


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]


setup(
    name="decision-tree",
    python_requires=">=3.5.2",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    test_suite="nose.collector",
    tests_require=["mock", "nose"]
)
