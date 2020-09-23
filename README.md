# HOLMES

A Python package for inferring Higher-Order dependencies with a Log-linear Estimation Strategy from presence/absence data.  HOLMES is also used for generating synthetic presence/absence from known higher-order structures.  

The codes are based on methods developed in Ref. 1.  These methods rely on log-linear models, which are classical statistical tools of discrete multivariate analysis (Ref. 2), and two mathematical structures that generalize the concept of graph, namely simplicial complexes and hypergraphs (Ref. 3).  

## Table of content

1. [Usage](#usage)
    1. [Dependencies](#dependencies)
    2. [Installation](#installation)
    3. [Examples](#examples)
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
```

### Examples

* To analyze existing presence/absence data: `$ cd data_analysis` and look at the script `example1.py` 
* To generate synthetic data: `$ cd generative_model` and open the script `example2.py` 

## References

1. Xavier Roy-Pomerleau (2020). Inférence d'interactions d'ordre supérieur et de complexes simpliciaux à partir de données de présence/absence.  Master's thesis.  Université Laval. PDF available [here](https://dynamicalab.github.io/assets/pdf/theses/Roy-Pomerleau20_master.pdf)
2. Yvonne M. Bishop, Stephen E. Fienberg, and Paul W. Holland (2007). Discrete Multivariate Analysis: Theory and Practice. Springer.
3. Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora, Maxime Lucas, Alice Patania, Jean-Gabriel Young, Giovanni Petri (2020). Networks beyond pairwise interactions: structure and dynamics. [Physics Reports 874, 25, 1-92.](https://doi.org/10.1016/j.physrep.2020.05.004)
