#https://docs.python.org/2/distutils/index.html
#to find all available distribution formats:
#    $ python setup.py bdist --help-formats


# PROBLEM: after gunzip, tar -xf, & python builds build/lib/eugene-0.1-0 ...
#          python tries to put the contents of eugene (eugene/*) instide
#          inside python2/site-packs/
# GOAL: get python to dump simply eugene, a directory, in site-packs
# KIM: how will this affect the use of "import eugene" statement?

from distutils.core import setup #,Extension

#directory_created()
#file_created()


setup(name='eugene', 

      #format: major.minor[.patch][sub]
      version='0.1.0', 

      author='Ben Jantzen',
      author_email='bjantzen@vt.edu',
      maintainer='...JP?',
      maintainer_email='jpgo5@vt.edu',
      url='http://www.ratiocination.org/',
      download_url='',
      description='AI for automated scientific discovery',
      long_description='',
      requires=['numpy', 'scipy'],
      #KIM: which versions are valid?
      classifiers=['Development Status :: 0 - Beta',
                   'Environment :: Console?',
                   'Intended Audience :: Developers & Scientists',
                   'License :: ...',
                   'Operating System :: Linux :: ArchLinux',
                   'Operating System :: Linux :: Debian Linux',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: Python',
                   'Topic :: AI - automated scientific discovery',],
      platforms='',
      license='',


      #specifying package as 'eugene' instead of root ('') installs
      #the package as a folder, rather than dumping the contents of that folder

      packages=['eugene', 'eugene.src', 
                'eugene.src.connect', 
                'eugene.src.virtual_sys',
                'eugene.tests',
                'eugene.scripts_demos']
#creates
#         site-packages/eugene/{src/, tests/}
#   and   site-packages/eugene-0.1.0-py2.7.egg-info/

)

