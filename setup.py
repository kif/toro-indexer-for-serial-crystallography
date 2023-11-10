from setuptools import setup, find_packages

VERSION = '1.0'
DESCRIPTION = 'TORO SX'
LONG_DESCRIPTION = 'Torch-Powered Robust Optimization for Serial Crystallography'

# Setting up
setup(
    # the name must match the folder name 'toro'
    name="toro",
    version=VERSION,
    author="Luis Barba",
    author_email="<luis.barba-flores@psi.ch>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
)