"""
Example module to test LGP with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.gp.tiny_lgp import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var, Const

try:
    from icecream import ic
except ModuleNotFoundError:
    ic = lambda *args, **kwargs: None

ARITHMETIC_FUNCTIONS = [ADD, SUB, MUL, DIV]


def main():
    functions = ARITHMETIC_FUNCTIONS
    terminals = [Var(0), Const(1)]

    hyperparameters = LGPHyperparameters(
        mu=1000,
        probability_mutation=0.3,
        branch_probability=0.0,
        p_register = 1,
        max_len = 30
    )
    config = LGPConfig(
        num_jobs=1,
        max_generations=50_000 - hyperparameters.mu,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=1000,
        max_time=500,
        num_registers=5,
        global_seed=13,
        checkpoint_interval=100,
        checkpoint_dir="checkpoints",
        experiment_name="my_experiment",
    )

    loss = absolute_distance
    data, actual = SRBenchmark().generate("KOZA3")
    problem = BlackBox(data, actual, loss, 1e-6, True)

    lgp = TinyLGP(functions, terminals, config, hyperparameters)
    lgp.evolve(problem)


if __name__ == "__main__":
    main()
