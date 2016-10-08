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

      author='Benjamin C. Jantzen',
      author_email='bjantzen@vt.edu',
      maintainer='Benjamin C. Jantzen',
      maintainer_email='bjantzen@vt.edu',
      url='http://www.ratiocination.org/',
      download_url='https://github.com/jantzen/eugene', 
      description='tools for automated scientific discovery',
      requires=['numpy', 'scipy'],

      classifiers=['Development Status :: 1 - Dev',
                   'Intended Audience :: Developers & Scientists',
                   'License :: ...',
                   'Programming Language :: Python',
                   'Topic :: automated scientific discovery',],
      platforms='',
      license='',

      #specifying package as 'eugene' instead of root ('') installs
      #the package as a folder, rather than dumping the contents of that folder

      packages=['eugene', 'eugene.src', 
                'eugene.src.connect', 
                'eugene.src.virtual_sys',
                'eugene.tests',
                'eugene.scripts_demos']
)

