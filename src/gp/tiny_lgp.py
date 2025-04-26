"""
TinyLGP: A minimalistic implementation of Linear Genetic Programming for
         TinyverseGP.

         Genome representation:
         Mutation operator:
         Search algorithm:
"""

from typing import Sequence, Any

import math
import random
import time
import operator
from dataclasses import dataclass
from collections import namedtuple
from src.gp.functions import Function
from src.gp.tinyverse import GPModel, Hyperparameters, GPConfig

try:
    from icecream import ic
except ModuleNotFoundError:
    ic = lambda *args, **kwargs: None

LGP_CONDITIONS = [
    Function(2, 'Test_LT', operator.lt),
    Function(2, 'Test_GT', operator.gt),
    Function(2, 'Test_EQ', operator.eq),
    Function(2, 'Test_NE', operator.ne),
]


@dataclass
class LGPHyperparameters(Hyperparameters):
    """
    Specialized hyperparameter configuration space for LGP.
    """

    mu: int
    lambda_: int
    # levels_back: int
    # strict_selection: bool
    # mutation_rate: float = None
    # mutation_rate_genes: int = None

    def __init__(self, *, mu=10, lambda_=20, **kwargs):
        self.mu = mu
        self.lambda_ = lambda_
        super().__init__(**kwargs)


@dataclass
class LGPConfig(GPConfig):
    """
    Specialized GP configuration that is needed to run LGP.
    """

    num_inputs: int
    num_registers: int
    # num_function_nodes: int
    # num_functions: int
    # max_arity: int
    max_time: int
    report_every_improvement: bool = False

    def __init__(
        self,
        *,
        num_inputs,
        num_outputs=1,
        num_registers=16,
        report_every_improvement=True,
        **kwargs,
    ):
        self.num_inputs = num_inputs
        self.num_registers = num_registers
        self.report_every_improvement = report_every_improvement
        super().__init__(**kwargs, num_outputs=num_outputs)


Instruction = namedtuple('Instruction', ['dest', 'operator', 'operands'])


@dataclass
class LGPIndividual:
    """
    Class that is used to represent a LGP individual.
    Formally ...
    """

    genome: tuple[Instruction]
    fitness: float | None = None

    def __str__(self):
        return f"len={len(self.genome)},fit={self.fitness}"

    def __repr__(self):
        return str(self)


class TinyLGP(GPModel):
    """
    Main class of the tiny CGP module that derives from GPModel and
    implements all related fundamental mechanisms tun run CGP.
    """

    num_evaluations: int
    population: list[LGPIndividual]
    config: LGPConfig
    hyperparameters: LGPHyperparameters
    functions: Sequence[Function]

    def __init__(
        self,
        problem,
        functions: Sequence[Function],
        config: LGPConfig,
        hyperparameters: LGPHyperparameters,
    ):
        self.num_evaluations = 0
        self.problem = problem
        self.functions = tuple(functions)
        self.config = config
        self.hyperparameters = hyperparameters

    def _create_constant(self):
        return random.random() * 2 - 1

    def _create_random_genome(self, min_len=5, max_len=10, p_register=0.5) -> tuple[Instruction]:
        genome = list()
        possible_destinations = [f'R{n}' for n in range(self.config.num_registers)]
        possible_operands = possible_destinations + [
            f'I{n}' for n in range(len(self.problem.observations[0]))
        ]
        for i in range(random.randint(min_len, max_len)):
            dest = random.choice(possible_destinations)
            operator = random.choice(self.functions)
            operands = [
                (
                    random.choice(possible_operands)
                    if random.random() < p_register
                    else self._create_constant()
                )
                for _ in range(operator.arity)
            ]
            genome.append(Instruction(dest, operator, operands))
        return genome

    def evolve(self) -> Any:
        self.population = [
            LGPIndividual(self._create_random_genome(), None)
            for i in range(self.hyperparameters.mu)
        ]
        self.evaluate()
        ic(self.population)

    def predict(self, genome: Sequence[Instruction], observation: list):
        registers = {f'R{n}': 0.0 for n in range(10)} | {
            f'I{n}': v for n, v in enumerate(observation)
        }
        for instruction in genome:
            # if the operand is the name of a register use the value in the register, otherwise use it diretly
            registers[instruction.dest] = instruction.operator(
                *[registers[r] if r in registers else r for r in instruction.operands]
            )
        return [registers['R0']]

    def expression(self, genome: Sequence[Instruction]) -> Any:
        return '\n'.join(
            f'''{i.dest} = {i.operator.name}({', '.join(str(_) for _ in i.operands)})'''
            for i in genome
        )

    def fitness(self, individual: LGPIndividual) -> float:
        """
        Return the fitness value of an individual.
        """
        return individual.fitness

    def evaluate(self) -> float:
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_individual(individual.genome)
        if self.config.minimizing_fitness:
            self.population.sort(key=lambda i: i.fitness)
        else:
            self.population.sort(key=lambda i: i.fitness, reverse=True)
        return self.population[0].fitness

    def evaluate_individual(self, genome: Sequence[Instruction]) -> float:
        """
        Evaluates an individual against the problem.

        :param genome: the genome of an individual
        :return: fitness of the individual
        """
        self.num_evaluations += 1
        return self.problem.evaluate(genome, self)
