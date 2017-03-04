""" Setup for distribution of EUGENE source code.
"""

from setuptools import setup, find_packages


# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='eugene', 

      version='0.1.0',

      description='Tools for automated scientific discovery',
      long_description=long_description,

      url='http://www.ratiocination.org/',

      author='Benjamin C. Jantzen',
      author_email='bjantzen@vt.edu',

      maintainer='Benjamin C. Jantzen',
      maintainer_email='bjantzen@vt.edu',

      download_url='https://github.com/jantzen/eugene', 

      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence', 
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
                   ],

      keywords='discovery kinds dynamical system-identification natural-kinds'

      #specifying package as 'eugene' instead of root ('') installs
      #the package as a folder, rather than dumping the contents of that folder

      install_requires=['numpy', 'scipy'],

      packages=['eugene', 'eugene.src', 
                'eugene.src.connect', 
                'eugene.src.virtual_sys',
                'eugene.tests',
                'eugene.scripts_demos']
)

