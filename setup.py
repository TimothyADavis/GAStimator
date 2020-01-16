from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

 
setup(name='gastimator',
       version='0.1.9',
       description='Implementation of a Python MCMC gibbs-sampler with adaptive stepping',
       url='https://github.com/TimothyADavis/GAStimator',
       author='Timothy A. Davis',
       author_email='DavisT@cardiff.ac.uk',
       long_description=long_description,
       long_description_content_type="text/markdown",
       license='GNU GPLv3',
       packages=['gastimator'],
       install_requires=[
           'numpy',
           'matplotlib',
           'progress',
           'plotbin',
       ],
       classifiers=[
         'Development Status :: 3 - Alpha',
         'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
       ],
       zip_safe=False)
       
