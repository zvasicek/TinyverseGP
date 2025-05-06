"""
Example module to test GE with program synthesis problems.

Attempts to evolve a solution for the numbers of two problem that
is provided on Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

"""

import warnings
warnings.filterwarnings("ignore")

from src.gp.tiny_cgp import *
from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.gp.functions import *
from src.gp.tiny_ge import *


config = GPConfig(
    num_jobs=1,
    max_generations=1000,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60
)

hyperparameters = GEHyperparameters(
    pop_size=100,
    genome_length=40,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2
)

generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n,m])
problem = ProgramSynthesis(benchmark.dataset)

functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
arguments = ['x']
grammar = {
    '<expr>': [
        'ADD(<expr>, <expr>)', 'SUB(<expr>, <expr>)', 'MUL(<expr>, <expr>)', 'DIV(<expr>, <expr>)',
        'AND(<expr>, <expr>)', 'OR(<expr>, <expr>)', 'NAND(<expr>, <expr>)', 'NOR(<expr>, <expr>)', 
        'NOT(<expr>)', 'IF(<expr>, <expr>, <expr>)', 'LT(<expr>, <expr>)', 'GT(<expr>, <expr>)',
        '<d>', '<d>.<d><d>', 'x'
    ],
    '<d>': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
}

ge = TinyGE(problem, functions, grammar, arguments, config, hyperparameters)
ge.evolve()