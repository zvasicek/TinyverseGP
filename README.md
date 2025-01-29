# TinyverseGP: minimalistic implementations of different representations for Genetic Programming 

TinyverseGP is a collection of minimalistic implementations of different representations for Genetic Programming. The goal is to provide a simple and easy-to-understand codebase with the following goals in mind:

- **Minimalistic**: The codebase should be as small as possible, while still being able to demonstrate the core concepts of the representation.
- **Educational**: The codebase should be easy to understand and serve as a starting point for learning about Genetic Programming.
- **Extensible**: The codebase should be easy to extend and modify, so that it can be used as a basis for further research and experimentation.
- **Benchmarking**: The codebase should be able to run on standard benchmark problems and datasets, so that it can be used to compare different representations and algorithms.

The codebase is written in Python trying to keep the requirements to a minimal. The codebase is organized into different modules, each of which implements a different representation for Genetic Programming. The following representations are currently implemented:

- Tree-based Genetic Programming (TGP) (also known as Koza-style): the programs are represented as trees. This version supports multi-tree chromosomes, where each individual is represented by a set of trees when the problem requires multiple outputs.
- Cartesian Genetic Programming (CGP): the programs are represented graphs and naturally encodes multiple outputs with shared components.

This repository is organized as follows:

- `src/gp`: contains the core implementation of the different representations.
  - `tiny_tgp.py`: implementation of Tree-based Genetic Programming (TGP).
  - `tiny_cgp.py`: implementation of Cartesian Genetic Programming (CGP).
  - `tineverse.py`: the abstract classes for GP, Config, Hyperparameters, and Function set.
  - `functions.py`: the standard set of functions currently supported.
  - `problem.py`: the abstract class for the problem to be solved. It includes the example based problem (black-box), policy search, and program synthesis.
  - `loss.py`: currently supported loss functions.
- `src/benchmark`: contains the benchmark problems and datasets.
  - `symbolic_regression/sr_benchmark.py`: sample symbolic regression benchmark problems.
  - `symbolic_regression/srbench.py`:  interface to the SRBench benchmark suite.
  - `logic_synthesis/ls_benchmark.py`: sample logic synthesis benchmark problems.
  - `policy_search/policy_evaluation.py`: interface to the gymnasium environment.
- `src/examples`: examples on how to use the different benchmarks.

# Requirements and testing

The current version supports Python3.9 and higher. To install the requirements it is suggest to run:


```bash
python3 -m venv env
. env/bin/activate
pip3 install -r requirements.txt
```

To run the examples, you can use the following command:

```bash
python3 src/examples/test_tgp.py
```

or any other script in that folder.

# Contributing

The codebase is still in its early stages and contributions are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
