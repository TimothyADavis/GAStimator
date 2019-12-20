# GAStimator
Implementation of a Python MCMC gibbs-sampler with adaptive stepping. 

While this is a simple MCMC algorithm, it is robust and stable and well suited to high dimensional problems with many degrees of freedom and very sharp likelihood features. For instance kinematic modelling of datacubes with this code has been found to be orders of magnitude quicker with GAStimator than using more advanced affine-invariant MCMC methods. 

### Install
You can install GAStimator with `pip install gastimator`. Alternativly you can download the code here, nagivate to the directory you unpack it too, and run `python setup.py install`.
    
It requires the following modules:

* numpy
* matplotlib
* progress
* plotbin

### Documentation

To get you started, see the walk through here: https://github.com/TimothyADavis/GAStimator/blob/master/documentation/GAStimator_Documentation.ipynb


Author & License
-----------------

Copyright 2019 Timothy A. Davis

Built by `Timothy A. Davis <https://github.com/TimothyADavis>`. Licensed under
the GNU General Public License v3 (GPLv3) license (see ``LICENSE``).