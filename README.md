# TinyverseGP: Minimalistic implementations of different representations for Genetic Programming 

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

To run the examples, you can use one of the following command:

```bash
python3 -m examples.symbolic_regression.test_cgp_sr
python3 -m examples.symbolic_regression.test_tgp_sr
python3 -m examples.logic_synthesis.test_cgp_ls
python3 -m examples.logic_synthesis.test_tgp_ls
python3 -m examples.policy_learning.test_cgp_pl
python3 -m examples.policy_learning.test_cgp_pl_ale
python3 -m examples.policy_learning.test_tgp_pl
python3 -m examples.program_synthesis.test_cgp_ps
python3 -m examples.program_synthesis.test_tgp_ps
```

or any other script in that folder.

# Contributing

This repository is kept under a Github Organization to allow for a more inviting environment for contributions. The organization will not be tied to any specific institution and will be open to all contributors. If you want to contribute, please contact the maintainers to be added to the organization as a maintainer.
The codebase is still in its early stages and contributions are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue, submit a pull request or open a new discussion.

## Creating a new representation module

To create a new representation, you can follow the following steps:

- Create a new Python script in the `src/gp` folder with the implementation of the representation. As a convention, name the script `tiny_<first letter of the representation>gp.py`. For example, `tiny_tgp.py` for Tree-based Genetic Programming and `tiny_cgp.py` for Cartesian Genetic Programming.
- Implement the representation as a class and create a `Tiny<first letter of the representation>GP` class that inherits from the `GPModel` class. The `GPModel` class is an abstract class that defines the interface for the different representations. This class should contain the following fields:

- config: the configuration class inherited from Config abstract class.
- hyperparameters: the hyperparameters class inherited from Hyperparameters abstract class.
- problem: the problem class.
- functions: a list of functions (non-terminals)

- Implement the following methods in the `Tiny<first letter of the representation>GP` class:

  - `fitness(self, individual)`: the fitness function of a single individual.
  - `evolve(self)`: the evolution method that evolves the population.
  - `selection(self)`: the selection method that selects individuals for recombination and perturbation.
  - `predict(self, genome, observation)`: the prediction method that predicts the output of `genome` to a single `observation`.
  - `expression(self, genome)`: the expression method that returns the expression represented by `genome`.

## Creating a new problem domain module

To create a new problem domain, you can follow the following steps:

- Update the file `src/gp/problem.py` with a new class that inherits from the `Problem` abstract class. This class should contain the following methods:
  - `is_ideal(self, fitness)`: a method that returns True if the fitness reached an ideal state (i.e., known optima).
  - `is_better(self, fitness1, fitness2)`: a method that returns True if `fitness1` is better than `fitness2`.
  - `evaluate(self, genome, GPModel)`: a method that instructs how to evaluate a given `genome` using a `GPModel`

A good starting point is to look at the `BlackBox` and `PolicySearch` classes in the `problem.py` file which gives examples of two very different problem domains.

Finally, if you want to create an interface to an existing benchmark suite, you can look at the examples in:
- `src/benchmark/symbolic_regression/srbench.py`: interface to the SRBench benchmark suite.

# Roadmap

See `Roadmap.md` for the current roadmap.

# Collaborators

See `Collaborators.md` for the current list of collaborators. Please, update this file after pull requests are merged describing your collaboration.

# Citing 

TBD 

# LICENSE

This work is under GNU General Public License, Version 3.

# Acknowledgements

This work was supported b an Alexander von Humboldt Professorship in AI held by Holger Hoos, the Czech Science Foundation project 25-15490S and Conselho Nacional de Desenvolvimento Cient\'{i}fico e Tecnol\'{o}gico (CNPq) grant 301596/2022-0.
