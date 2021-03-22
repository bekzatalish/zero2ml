"""
Setup file for the package.
"""
import os
import sys
import zero2ml
from setuptools import find_packages, setup


# Python version check
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

here = os.path.abspath(os.path.dirname(__file__))

# Helper function to read textfiles
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
NAME = "zero2ml"
VERSION = zero2ml.__version__
DESCRIPTION = zero2ml.__description__
LONG_DESCRIPTION = read("README.md")
URL = zero2ml.__url__
AUTHOR = zero2ml.__author__
AUTHOR_EMAIL = zero2ml.__email__
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
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    license=LICENSE,
    python_requires=">=3.6",
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    zip_safe=False
)
