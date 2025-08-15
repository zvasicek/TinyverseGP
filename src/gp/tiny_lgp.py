# This file is part of TinyverseGP.
#
# TinyverseGP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TinyverseGP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TinyverseGP.  If not, see <https://www.gnu.org/licenses/>.
#
# Algorithm based on the paper: https://link.springer.com/book/10.1007/978-0-387-31030-5

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
import copy
from dataclasses import dataclass
from collections import namedtuple
from src.gp.functions import Function
#from src.gp.tinyverse import GPModel, Hyperparameters, GPConfig
from src.gp.problem import *
from src.gp.tinyverse import *

try:
    from icecream import ic
except ModuleNotFoundError:
    ic = lambda *args, **kwargs: None

LGP_CONDITIONS = [
    Function(2, "Test_LT", operator.lt),
    Function(2, "Test_GT", operator.gt),
    Function(2, "Test_EQ", operator.eq),
    Function(2, "Test_NE", operator.ne),
]


@dataclass(kw_only=True)
class LGPHyperparameters(Hyperparameters):
    """
    Specialized hyperparameter configuration space for LGP.
    """

    mu: int
    tournament_size: int
    min_len : int 
    max_len : int 
    p_register : float 
    probability_mutation : float
    branch_probability : float
    erc : bool
    default_value : float
    # levels_back: int
    # strict_selection: bool
    # mutation_rate: float = None
    # mutation_rate_genes: int = None

    def __init__(self, *, mu=10, tournament_size=2, min_len = 5, max_len = 10, p_register = 0.5, probability_mutation = 0.1, branch_probability = 0.0, erc=False, default_value=0.0, **kwargs):
        self.mu = mu
        self.tournament_size = tournament_size
        self.min_len = min_len 
        self.max_len = max_len 
        self.p_register = p_register
        self.probability_mutation = probability_mutation
        self.branch_probability = branch_probability
        self.erc = erc
        self.default_value = default_value
        super().__init__(**kwargs)
    def __post_init__(self):
        Hyperparameters.__post_init__(self)


@dataclass(kw_only=True)
class LGPConfig(GPConfig):
    """
    Specialized GP configuration that is needed to run LGP.
    """

    num_registers: int
    # num_function_nodes: int
    # num_functions: int
    # max_arity: int
    max_time: int
    report_every_improvement: bool = False

    def __init__(
        self,
        *,
        num_outputs=1,
        num_registers=16,
        report_every_improvement=True,
        **kwargs,
    ):
        self.num_registers = num_registers
        self.report_every_improvement = report_every_improvement
        super().__init__(**kwargs, num_outputs=num_outputs)
    def __post_init__(self):
        GPConfig.__post_init__(self)


Instruction = namedtuple("Instruction", ["dest", "operator", "operands"])


@dataclass
class LGPIndividual:
    """
    Class that is used to represent a LGP individual.
    Formally ...
    """

    genome: tuple[Instruction]
    fitness: float | None = None

    def __str__(self):
        return f"0x{id(self):x}:l={len(self.genome)},f={self.fitness}"

    def __repr__(self):
        return str(self)

    def serialize_genome(self):
        return self.genome 
    def deserialize_genome(self, genome_):
        self.genome = genome_


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
        functions: Sequence[Function],
        terminals: Sequence[Function],
        config: LGPConfig,
        hyperparameters: LGPHyperparameters,
    ):
        super().__init__(config, hyperparameters)
        self.num_evaluations = 0
        self.functions = tuple(functions)
        self.terminals = tuple(terminals)
        self.config = config
        self.hyperparameters = hyperparameters
        self.best_individual = None 
        self.num_evaluations = 0

        self.population = [LGPIndividual(self._create_random_genome(self.hyperparameters.min_len, self.hyperparameters.max_len), None) for _ in range(self.hyperparameters.mu)]

    def _create_constant(self):
        return random.random() * 2 - 1

    def _create_random_genome(self, min_len=1, max_len=4) -> tuple[Instruction]:
        genome = list()
        read_write = [f"R{n}" for n in range(self.config.num_registers)]
        read_only = [f"I{n}" for n in range(len(self.terminals))]
        possible_operands = read_write + read_only

        for i in range(random.randint(min_len, max_len)):
            if random.random() < self.hyperparameters.branch_probability:
                dest = None
                operator = random.choice(LGP_CONDITIONS)
            else:
                dest = random.choice(read_write)
                operator = random.choice(self.functions)

            operands = []
            can_add_read = operator.arity > 1
            for _ in range(operator.arity):
                if can_add_read and random.random() <= self.hyperparameters.p_register:
                    operands.append(random.choice(read_write))
                else:
                    if self.hyperparameters.erc and random.random() < 0.5:
                        operands.append(self._create_constant())
                    else:
                        operands.append(random.choice(read_only))
                    can_add_read = False
            genome.append(Instruction(dest, operator, operands))
        return genome

    def breed(self, problem):
        w1, l1 = self.tournament_selection(problem.minimizing)
        w2, l2 = self.tournament_selection(problem.minimizing)
        offspring1, offspring2 = self.crossover(
            self.population[w1], self.population[w2]
        )
        if random.random() < self.hyperparameters.probability_mutation:
            self.mutate(offspring1)
            self.mutate(offspring2)

        offspring1.fitness = self.penalize(self.evaluate_individual(offspring1.genome, problem), offspring1.genome)
        offspring2.fitness = self.penalize(self.evaluate_individual(offspring2.genome, problem), offspring2.genome)
        # SURVIVAL
        tmp = [offspring1, self.population[l1]]
        self._sort(tmp, problem.minimizing)
        self.population[l1] = tmp[0]
        tmp = [offspring2, self.population[l2]]
        self._sort(tmp, problem.minimizing)
        self.population[l2] = tmp[0]

        self._sort(self.population, problem.minimizing)
        best = copy.copy(self.population[0])
        best_fitness = best.fitness

        self.best_individual = copy.copy(best)

        #print(self.expression(best.genome), best.fitness)

        return self.population[0]

    def mutate(self, individual: LGPIndividual) -> LGPIndividual:
        pos = random.randint(0, len(individual.genome) - 1)
        # 1: cut instruction, 2: insert, 3: change destiny, 4: change operator, 5: change operand
        muts = [1,2,3,4,5] if len(individual.genome) > 1 else [2,3,4,5]
        p = random.choice(muts)
        read_write = [f"R{n}" for n in range(self.config.num_registers)]
        read_only = [f"I{n}" for n in range(len(self.terminals))]
        if p==1: # cut off
            return LGPIndividual(
                individual.genome[:pos] + individual.genome[pos + 1 :], None
            )
        elif p==2: # insert new
            instruction = self._create_random_genome(min_len=1, max_len=1)
            return LGPIndividual(
                individual.genome[:pos] + instruction + individual.genome[pos:], None
            )
        elif p==3:
            dest, operator, operands = individual.genome[pos]
            individual.genome[pos] = Instruction(random.choice(read_write), operator, operands)
            return LGPIndividual(
                individual.genome, None
            )
        elif p==4:
            dest, operator, operands = individual.genome[pos]
            arity = operator.arity
            operator = random.choice([f for f in self.functions if f.arity == arity])
            individual.genome[pos] = Instruction(dest, operator, operands)
            return LGPIndividual(
                individual.genome, None
            )
        elif p==5:
            op = random.randint(0, len(individual.genome[pos].operands)-1)
            operand = individual.genome[pos].operands[op]
            dest, operator, operands = individual.genome[pos]
            if isinstance(operand, str):
                if operand[0] == "I":
                    operands[op] = random.choice(read_only + read_write)
                else:
                    operands[op] = random.choice(read_write)
            else:
                operands[op] = self._create_constant()
            individual.genome[pos] = Instruction(dest, operator, operands)
            return LGPIndividual(
                individual.genome, None
            )

    def crossover(
        self, individual1: LGPIndividual, individual2: LGPIndividual
    ) -> tuple[LGPIndividual, LGPIndividual]:
        cut1_1 = random.randint(0, max(0, len(individual1.genome) - 2))
        cut1_2 = random.randint(cut1_1, max(0, len(individual1.genome) - 1))
        cut2_1 = random.randint(0, max(0, len(individual2.genome) - 2))
        cut2_2 = random.randint(cut2_1, max(0, len(individual2.genome) - 1))
        n1 = cut1_2 - cut1_1 + 1
        n2 = cut2_2 - cut2_1 + 1
        offspring1 = copy.copy(individual1.genome[:cut1_1]) + copy.copy(individual2.genome[cut2_1:n2]) + copy.copy(individual1.genome[cut1_2:])
        offspring2 = copy.copy(individual2.genome[:cut2_1]) + copy.copy(individual1.genome[cut1_1:n1]) + copy.copy(individual2.genome[cut2_2:])

        if len(offspring1) == 0:
            offspring1 = copy.copy(individual1.genome)
        if len(offspring2) == 0:
            offspring2 = copy.copy(individual2.genome)
        return LGPIndividual(offspring1, None), LGPIndividual(offspring2, None)

    def predict(self, genome: Sequence[Instruction], observation: list):
        registerVars = { f"I{n}" : v.function() if v.const else observation[v.function()] for n, v in enumerate(self.terminals) }
        registers = {f"R{n}": self.hyperparameters.default_value for n in range(self.config.num_registers)} | registerVars
        skip_next = False
        for instruction in genome:
            if skip_next:
                skip_next = False
            else:
                value = instruction.operator.function(
                    *[
                        registers[r] if isinstance(r, str) else r
                        for r in instruction.operands
                    ]
                )
                if instruction.dest is not None:
                    registers[instruction.dest] = value
                elif not value:
                    skip_next = True
        return [registers[f"R{i}"] for i in range(self.config.num_outputs)]

    def expression(self, genome: Sequence[Instruction]) -> Any:
        return "\n".join(
            f"""{i.dest} = {i.operator.name}({', '.join(str(_) for _ in i.operands)})"""
            for i in genome
        )

    def fitness(self, individual: LGPIndividual) -> float:
        """
        Return the fitness value of an individual.
        """
        return individual.fitness

    '''
    def evaluate(self) -> float:
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_individual(individual.genome)
        self._sort(self.population)
        return self.population[0].fitness
    '''
    def _sort(self, individuals: list[LGPIndividual], minimizing_fitness):
        individuals.sort(
            key=lambda i: i.fitness, reverse=not minimizing_fitness
        )

    def tournament_selection(self, minimizing_fitness) -> tuple[int, int]:
        """Return indexes of winner and loser"""
        for i, individual in enumerate(self.population):
            individual.idx = i
        candidates = random.choices(
            self.population, k=self.hyperparameters.tournament_size
        )
        self._sort(candidates, minimizing_fitness)
        return candidates[0].idx, candidates[-1].idx

    def evaluate_individual(self, genome: Sequence[Instruction], problem) -> float:
        """
        Evaluates an individual against the problem.

        :param genome: the genome of an individual
        :return: fitness of the individual
        """
        self.num_evaluations += 1
        f = problem.evaluate(
            genome, self
        )  # evaluate the solution using the problem instance
        if self.best_individual is None or problem.is_better(
            f, self.best_individual.fitness
        ):
            self.best_individual = LGPIndividual(genome, f)
        return problem.evaluate(genome, self)

    def eval_complexity(self, genome: list[int]) -> float:
        """
        Returns the complexity of the genome based on the number of active nodes.

        :param genome: Genome of an individual
        :return: Complexity value
        """
        nodes = 0 
        for instruction in genome:
            nodes += len(instruction.operands) + 1
        return nodes

    def is_valid(self, genome: list[int]) -> bool:
        """ """
        return True

    def pipeline(self, problem):
        """
        Pipeline that performs one generational step CGP in the common
        1+lambda fashion.

        :return: best solution found in the population
        """
        return self.breed(problem)

    def selection(self) -> list:
        """
        Performs a 1 + lambda strategy with either
        strict or non-strict selection. Non-strict selection
        allows to explore the neutral neighbourhood of the
        parent which has been found to be very effective for
        the use of CGP:

        :return: parent individual
        """
        raise NotImplementedError
