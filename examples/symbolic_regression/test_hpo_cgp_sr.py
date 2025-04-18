"""
Example module to test CGP with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.gp.tiny_cgp import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var, Const
from src.hpo.hpo import Hpo

functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

config = CGPConfig(
    num_jobs=1,
    max_generations=10,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    num_function_nodes=10,
    report_interval=1,
    max_time=60
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True
)
config.init()

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate('KOZA3')

problem = BlackBox(data, actual, loss, 1e-6, True)

cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)

## hpo with smac, test
opt_hyperparameters = Hpo.optimise_smac(cgp)
print(opt_hyperparameters)
#cgp.hyperparameters = opt_hyperparameters
#cgp.evolve()
