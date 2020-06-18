# sqtom - Squeezed state tomography in Python

![GitHub Workflow Status (master)](https://img.shields.io/github/workflow/status/XanaduAI/sqtom/Tests/master?style=flat-square)
![Codecov coverage](https://img.shields.io/codecov/c/github/xanaduai/sqtom/master.svg?style=flat-square)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/XanaduAI/sqtom/master?style=flat-square)
![PyPI](https://img.shields.io/pypi/v/sqtom.svg?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqtom.svg?style=flat-square)

This repository implements the mode tomography ideas presented in

*"Full statistical mode reconstruction of a light field via a photon-number-resolved measurement"*
by Burenkov. et al. [Phys. Rev. A 95, 053806 (2017)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.053806)
and in Burenkov et al. in [J. Res. Natl. Inst. Stan. 122, 30 (2017)](https://doi.org/10.6028/jres.122.030).


for twin beam light and extends it to degenerate squeezed light. By leveraging `lmfit` we can also
give a number of uncertainty estimates, and moreover provide routines for thresholding photon-number
measurements and useful heuristics for initial guesses for the solutions of the problem.

## Contents

The main physical ideal used by Burenkov et al. is to model the joint photon distribution of the
variables associated to the photon numbers in signal and idler beams as resulting from one or
several lossy two-mode squeezed distributions hitting the detectors. To model dark counts they also
allow for modes prepared in states with Poisson statistics to hit the detectors.

To obtain the joint probability distribution of the photon numbers in the signal and idlers one
needs to *convolve* the probability distributions of the modes entering in the problem.

## Requirements

* [SciPy](https://www.scipy.org/) to calculate probability distributions of Poisson, Geometric or
  Negative Binomial random variables.

* [NumPy](https://numpy.org/) to perform 2D convolutions and matrix manipulations.

* [The Walrus](https://the-walrus.readthedocs.io/en/latest/) to calculate loss matrices and squeezed
  states probability distributions.

With the tools described so far we can solve the *forward* problem, i.e., given a set of physical
parameters what is the probability distribution.

* If we augment our tools with [lmfit](https://lmfit.github.io/lmfit-py/) we can solve the *inverse*
  problem: to find the best set of parameters that explain a given observed frequency distribution
  of photon numbers.

Finally, we use [pytest](https://docs.pytest.org/en/latest/) for testing.

All of these prerequisites can be installed via `pip`:

```bash
pip install sqtom
```

## Contributors

Nicolas Quesada

## License

This source code is free and open source, released under the Apache License, Version 2.0.
