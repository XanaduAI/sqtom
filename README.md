# sqtom
## Squeezed state tomography

This repository implements the mode tomography ideas presented in

*"Full statistical mode reconstruction of a light field via a photon-number-resolved measurement"* by Burenkov. et al. [Phys. Rev. A 95, 053806 (2017)
](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.053806)
"

for twin beam light and extends it to degenerate squeezed light.

See Examples.ipynb for examples.

## Contents

The main physical ideal used by Burenkov et al. is to model the joint photon distribution of the variables associated to the photon numbers in signal and idler beams as resulting from one or several lossy two-mode squeezed distributions hitting the detectors. To model dark counts they also allow for modes prepared in states with Poisson statistics to hit the detectors.

To obtain the joint probability distribution of the photon numbers in the signal and idlers one needs to *convolve* the probability distributions of the modes entering in the problem.

## Requirements

* [SciPy](https://www.scipy.org/) to calculate probability distributions of Poisson, Geometric or Negative Binomial random variables.

* [Numba](http://numba.pydata.org/) to rapidly calculate the stochastic matrices that model loss.

* [NumPy](https://numpy.org/) to perform 2D convolutions and matrix manipulations.

With the tools described so far we can solve the *forward* problem, i.e., given a set of physical parameters what is the probability distribution.

* If we augment out tools with [lmfit](https://lmfit.github.io/lmfit-py/) we can solve the *inverse* problem: to find the best of parameters that explain a given observed frequency distribution of photon numbers.

Finally, we used [pytest](https://docs.pytest.org/en/latest/) for testing.

All of these prerequisites can be installed via `pip`:

```bash
pip install numpy scipy numba lmfit pytest
```

## Authors



## License

This source code is free and open source, released under the Apache License, Version 2.0.
