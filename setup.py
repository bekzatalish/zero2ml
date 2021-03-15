"""Setup file for the package.
"""
import os
import zeroml
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    """
    Construct absolute path to given file, read the file, and return its contents.
    """
    with open(os.path.join(here, filename), encoding="utf-8") as f:
        file_content = f.read()
    return file_content

# Get dependencies from requirements.txt
INSTALL_REQUIRES = read("requirements.txt").split("\n")
INSTALL_REQUIRES = [x.rstrip().rstrip() for x in INSTALL_REQUIRES if x.rstrip() != ""]

# Core package components and metadata
NAME = "zero-ml"
VERSION = zeroml.__version__
DESCRIPTION = zeroml.__description__
LONG_DESCRIPTION = read("README.md")
URL = zeroml.__url__
AUTHOR = zeroml.__author__
AUTHER_EMAIL = zeroml.__email__
LICENSE = read("LICENSE")
PACKAGES = find_packages()

# Build script
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHER_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHER_EMAIL,
    license=LICENSE,
    python_requires=">=3.6",
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    zip_safe=False
)
