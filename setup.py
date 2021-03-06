from setuptools import setup, find_packages


NAME = 'supertensortools'
DESCRIPTION = 'Supervised Tools for Tensor Decomposition.'
AUTHOR = 'Alex Williams'
EMAIL = 'alex.h.willia@gmail.com'
VERSION = "0.1"
URL = 'https://github.com/ahwillia/supertensortools'
LICENSE = 'MIT'

install_requires = [
    'numpy',
    'scipy',
    'torch',
    'torchvision'
]
tests_require = ['pytest'] + install_requires
setup_requires = ['pytest-runner']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    python_requires='>=3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='tensor decomposition, canonical decomposition, parallel factors',
    packages=find_packages(exclude=['tests*']),
)
