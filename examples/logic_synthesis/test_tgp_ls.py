"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""

from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_tgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox

benchmark = LSBenchmark('../../data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs

functions = [AND, OR, NAND, NOR, NOT]
terminals = [Var(0)]

config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
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

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.get_outputs
problem = BlackBox(data, actual, loss, 0, True)

tgp = TinyTGP(problem, functions, terminals, config, hyperparameters)
tgp.evolve()
