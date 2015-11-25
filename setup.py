import os

from distutils.core import setup

packages = ["binoculars",
            "binoculars.backends"]

scripts = [os.path.join("scripts", d)
           for d in ["binoculars-fitaid",
                     "binoculars-gui",
                     "binoculars-processgui",
                     "binoculars"]]

setup(name='binoculars', version='0.0.1',
      description='FIXME',
      long_description='FIXME',
      packages=packages,
      scripts=scripts,
      author="Willem Onderwaater, Sander Roobol",
      author_email="onderwaa@esrf.fr",
      url='FIXME',
      license='GPL-3',
      classifiers=[
          'Topic :: Scientific/Engineering',
          'Development Status :: 3 - Alpha',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7']
      )
