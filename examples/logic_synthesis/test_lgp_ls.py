"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""

from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.tiny_lgp import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox

benchmark = LSBenchmark('data/logic_synthesis/plu/alu2.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs
print(num_inputs, num_outputs)

functions = [AND, OR, NAND, NOR, XOR, NOT]
terminals = [Var(i) for i in range(num_inputs)]

hyperparameters = LGPHyperparameters(
    mu=1000,
    macro_variation_rate=0.75,
    micro_variation_rate=0.25,
    insertion_rate=0.4,
    max_segment=15,
    reproduction_rate=0.5,
    branch_probability=0.0,
    p_register = 0.25,
    max_len = 100,
    initial_max_len = 50,
    erc = False,
    default_value = 1,
    protection = 1e10,
    penalization_validity_factor=200.0
)
config = LGPConfig(
        num_jobs=1,
        max_generations=100_000, # - hyperparameters.mu,
        stopping_criteria=0,
        minimizing_fitness=True,
        ideal_fitness=0,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=hyperparameters.mu,
        max_time=500,
        num_outputs=num_outputs,
        num_registers=12,
        global_seed=None,
        checkpoint_interval=1000000000,
        checkpoint_dir="checkpoints",
        experiment_name="logic_lgp",
)

loss = hamming_distance_bitwise
data = truth_table.inputs
actual = truth_table.outputs
problem = BlackBox(data, actual, loss, 0, True)

lgp = TinyLGP(functions, terminals, config, hyperparameters)
lgp.evolve(problem)
print(lgp.expression(lgp.best_individual.genome))
