# HOLMES

HOLMES stands for Higher-Order dependencies with a Log-linear Estimation Strategy. It is a collection of Python modules that are used for inferring higher-order dependencies or interactions from presence/absence data.  HOLMES is also used for randomly generating synthetic presence/absence data from known higher-order structures.  

The codes are based on the methods developed in Ref. 1.  These methods rely on log-linear models, which are classical statistical tools for discrete multivariate analysis (Ref. 2). The higher-order interactions (Ref. 3) are encoded into mathematical structures that generalize the concept of graph, namely simplicial complexes and hypergraphs.

## Table of content

1. [Usage](#usage)
    * [Dependencies](#dependencies)
    * [Installation](#installation)
    * [Examples](#examples)
2. [References](#references)

## Usage
### Dependencies

* `python3`
* `numpy`
* `numba`
 

### Installation

```
$ git clone https://github.com/pdesrosiers/HOLMES
$ cd HOLMES
$ python setup.py install
```

### Examples

* Some examples are provided in the folder `doc`.  
* The script `example1.py` explaines how to analyze existing presence/absence data with the asymptotic method
* The script `example2.py` generates synthetic presence/absence data from a list a facets defing a simplicial complex.

## References

1. Xavier Roy-Pomerleau (2020). Inférence d'interactions d'ordre supérieur et de complexes simpliciaux à partir de données de présence/absence.  Master's thesis.  Université Laval. PDF available [here](https://dynamicalab.github.io/assets/pdf/theses/Roy-Pomerleau20_master.pdf)
2. Yvonne M. Bishop, Stephen E. Fienberg, and Paul W. Holland (2007). Discrete Multivariate Analysis: Theory and Practice. Springer.
3. Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora, Maxime Lucas, Alice Patania, Jean-Gabriel Young, Giovanni Petri (2020). Networks beyond pairwise interactions: structure and dynamics. [Physics Reports 874, 25, 1-92.](https://doi.org/10.1016/j.physrep.2020.05.004)
