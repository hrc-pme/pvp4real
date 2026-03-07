import sys
from distutils.core import setup

from setuptools import find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name='pvp',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'pvp.sb3': ['version.txt', 'py.typed'],
    },
    include_package_data=True,
)
