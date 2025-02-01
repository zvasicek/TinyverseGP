"""
Example module to test TGP with symbolic regression problems.
"""

from src.gp.tiny_tgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark

config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60
)

hyperparameters = GPHyperparameters(
    pop_size=100,
    max_size=25,
    max_depth=5,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=2
)

loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate('KOZA1')
functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

problem = BlackBox(data, actual, loss, 1e-6, True)

tgp = TinyTGP(problem, functions, terminals, config, hyperparameters)
tgp.evolve()
