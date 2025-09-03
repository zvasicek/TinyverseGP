#!/usr/bin/env python3
from src.gp.tiny_cgp import *
from src.gp.problem import Problem
from src.gp.tinyverse import Var
import time
import requests

from dd.cudd import BDD
# import dd.cudd
#from dd.autoref import BDD, Function
#Function.__xor__ = lambda self, other: self._apply("xor", other)

from src.benchmark.logic_synthesis.blif_parser.blif import BlifFile
from src.gp.tinyverse import Function


AND = Function(2, "AND", lambda x, y: x & y)
OR = Function(2, "OR", lambda x, y: x | y)
NOT = Function(1, "NOT", lambda x: ~x)
NAND = Function(2, "NAND", lambda x, y: ~(x & y))
NOR = Function(2, "NOR", lambda x, y: ~(x | y))
XOR = Function(2, "XOR", lambda x, y: x ^ y)
XOR = Function(2, "XOR", lambda x, y: x.bdd.apply('xor', x, y))
XNOR = Function(2, "XNOR", lambda x, y: ~(x ^ y))
XNOR = Function(2, "XNOR", lambda x, y: ~x.bdd.apply('xor', x, y))
ID = Function(1, "ID", lambda x: x)


@dataclass
class BlackBoxBDD(Problem):
    """
    A black-box LS problem where the fitness is calculated by a loss function
    using a BDD representation of the target function.
    """

    def __init__(self, blif_: str, minimizing_: bool = True):
        self.blif = blif_
        self.minimizing = minimizing_
        self.ideal = 0

        known_gates = ["INVA", "IDA", "AND2", "OR2", "XOR2", "NAND2", "NOR2", "XNOR2"]
        b = BlifFile(known_gates=known_gates)
        self.num_inputs, self.num_outputs, gates = b.parse(blif_)

        self.bdd = BDD()

        vars = [g for g in b.eachInput()]
        self.input_variables = vars
        self.bdd.declare(*vars)

        g2v = {}
        self.bdd_vars = []
        for g in b.eachInput():
            v = self.bdd.var(g)
            g2v[g] = v
            self.bdd_vars.append(v)

        for g in b.eachGate():
            print(g, g.name, g.ina, g.inb, g.arity(), known_gates[g.fun], end=" ")

            ina = g2v[g.ina] if g.ina else None
            inb = g2v[g.inb] if g.inb else None

            # print(ina, inb)

            match known_gates[g.fun]:
                case "AND2":
                    o = ina & inb
                case "OR2":
                    o = ina | inb
                case "XOR2":
                    #o = ina ^ inb
                    o = self.bdd.apply('xor', ina, inb)
                case "NAND2":
                    o = ~(ina & inb)
                case "NOR2":
                    o = ~(ina | inb)
                case "XNOR2":
                    #o = ~(ina ^ inb)
                    o = ~self.bdd.apply('xor', ina, inb)
                case "INVA":
                    o = ~ina
                case "IDA":
                    o = ina
            g2v[g.name] = o
            #print(o)

        self.reference = []
        for o in b.eachOutput():
            print("output", o, g2v[o])
            self.reference.append(g2v[o])

        # sys.exit(0)
        # vars = [f'x{i}' for i in range(self.num_inputs)]
        # self.bdd_vars = [self.bdd.var(vars[i]) for i in range(self.num_inputs)]

    def evaluate(self, genome, model: GPModel) -> float:
        # print(genome)
        # print('decode', model.decode(genome))
        # print('active', model.active_nodes(genome))

        # Hamming distance using BDDs

        observation = self.bdd_vars
        prediction = model.predict_iter(genome, observation)

        hd = 0
        for i in range(self.num_outputs):
            odiff = self.bdd.apply("xor", prediction[i], self.reference[i])
            # odiff = prediction[i] ^ self.ref_vars[i]
            hd += odiff.count()
            # print('i', i, odiff, hd)
        return hd

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        """
        Check if the first fitness is better than the second.
        It takes into consideration whether the problem is minimizing or maximizing.
        """
        return fitness1 <= fitness2 if self.minimizing else fitness1 >= fitness2


functions = [NOT, ID, AND, OR, XOR, NAND, NOR, XNOR]
functions = [NOT, AND, OR, NAND, NOR]
functions = [ID, AND, XOR]
functions = [NOT, ID, AND, OR, XOR]

parity5 = """.model parity_5.blif
.inputs i0 i1 i2 i3 i4
.outputs out
.names i0 i1 s_05
10 1
01 1
.names i2 i3 s_06
10 1
01 1
.names s_05 s_06 s_07
10 1
01 1
.names s_07 i4 s_08
10 1
01 1
.names s_08 out
1 1
.end"""

problem = BlackBoxBDD(parity5)  # 'parity_5.blif')

#problem = BlackBoxBDD(
#    requests.get(
#        "https://raw.githubusercontent.com/boolean-function-benchmarks/benchmarks/refs/heads/main/benchmarks/blif/epar11.blif"
#    ).text
#)

# uncomment to evolve 3-bit adder
problem = BlackBoxBDD(
    requests.get(
        "https://raw.githubusercontent.com/boolean-function-benchmarks/benchmarks/refs/heads/main/benchmarks/blif/add3.blif"
    ).text
)
#
#problem = BlackBoxBDD(
#    requests.get(
#        "https://raw.githubusercontent.com/boolean-function-benchmarks/benchmarks/refs/heads/main/benchmarks/blif/mul3.blif"
#    ).text
#)

config = CGPConfig(
    num_jobs=1,
    max_generations=1_500_000,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=problem.num_inputs,
    num_outputs=problem.num_outputs,
    report_interval=5000,
    report_every_improvement=True,
    max_time=3600,
    global_seed=42,
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=4,
    population_size=5,
    levels_back=20,
    mutation_rate=0.05,
    num_function_nodes=30,
    #num_function_nodes=100,
    strict_selection=True,
)

#hyperparameters = CGPHyperparameters(
#    mu=1,
#    lmbda=4,
#    population_size=5,
#    levels_back=20,
#    mutation_rate=0.05,
#    num_function_nodes=50,
#    strict_selection=True,
#)

#config.init()
#random.seed(142)

# todo CONSTANTS
terminals = [Var(None, name_=n) for i, n in enumerate(problem.input_variables)]

data = None

cgp = TinyCGP(functions, terminals, config, hyperparameters)

print('NUM GENES', config.num_genes, config.num_genes*hyperparameters.mutation_rate)

tstart = time.time()
best = cgp.evolve(problem)
print(f'finished in {time.time()-tstart:.2f} seconds, generations {cgp.generation_number}')
print(f'best {best.fitness}, genome {best.genome}')
#print(cgp.evaluate_individual(best.genome))
print(f"expression {cgp.expression(best.genome)}")
