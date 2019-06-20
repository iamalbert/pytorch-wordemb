"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import os

here = path.abspath(path.dirname(__file__))

from torch.utils.cpp_extension import BuildExtension, CppExtension

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torchwordemb',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.9',

    description='Load pretrained word embeddings (word2vec, glove format) into torch.FloatTensor of PyTorch',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/iamalbert/pytorch-wordemb',

    # Author details
    author='WenLiZhuang',
    author_email='wlzhuang@nlg.csie.ntu.edu.tw',

    # Choose your license
    license='GPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='pytorch torch wordvectors nlp',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=['torchwordemb'],


    install_requires=[
        "torch>=1.0.0",
    ],
    setup_requires=[
        "torch>=1.0.0"
    ],

    ext_modules=[
        CppExtension(
            name='torchwordemb',
            sources=['src/loadwordemb.cpp'],
            extra_compile_args=["-Wall"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
