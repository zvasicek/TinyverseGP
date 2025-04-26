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
from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
from src.gp.tinyverse import GPModel, Hyperparameters, GPConfig, Var
from src.gp.problem import Problem

try:
    from icecream import ic
except ModuleNotFoundError:
    ic = lambda *args, **kwargs: None


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


class LGPIndividual:
    """
    Class that is used to represent a LGP individual.
    Formally ...
    """

    def __init__(self, genome: Sequence[Instruction], fitness: float | None = None):
        # TODO: Use multiple values for fitness
        self.genome = tuple(genome)
        self.fitness = fitness


class TinyLGP(GPModel):
    """
    Main class of the tiny CGP module that derives from GPModel and
    implements all related fundamental mechanisms tun run CGP.
    """

    num_evaluations: int
    population: list[LGPIndividual]

    def __init__(self):
        self.num_evaluations = 0

    def evolve(self) -> Any: ...

    def predict(self, genome: Sequence[Instruction], observations: list) -> list:
        predictions = list()
        for observation in observations:
            registers = {f'R{n}': 0.0 for n in range(10)} | {
                f'I{n}': v for n, v in enumerate(observation)
            }

            for instruction in genome:
                # if the operand is the name of a register, then use the value in the register, otherwise use it diretly
                registers[instruction.dest] = instruction.operator(
                    *[registers[r] if r in registers else r for r in instruction.operands]
                )
            predictions.append(registers['R0'])
        return predictions

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

    def evaluate(self) -> (LGPIndividual, bool):
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        best = None

    def evaluate_individual(self, genome: Sequence[Instruction]) -> float:
        """
        Evaluates an individual against the problem.

        :param genome: the genome of an individual
        :return: fitness of the individual
        """
        self.num_evaluations += 1
        return self.problem.evaluate(genome, self)
