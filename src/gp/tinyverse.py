"""
TinyverseGP: A modular cross-domain benchmark system for Genetic Programming.
             The tinyverse module includes a collections of base classes to be used by the GP modules.
             and therefore represent the fundamental model architecture of TinyverseGP.

             - GPModel: Abstract base class for GP representations
             - Config: Abstract base class for the GP configuration
                - GPConfig: Basic config class that considers fundamental
                            settings needed to run GP
             - Hyperparameters: Abstract base class for the hyperparameters
                - GPHyperparameters: Basic HP's often needed for the configuration of GP
             - Function: Base class for the representation of a non-terminal symbol
                         which is part of the function set
                    - Var:  Derived class for the representation of variable terminal symbols
                    - Const: Derived class for the representation of const terminal symbols
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any

class GPIndividual(ABC):
    genome: any
    fitness: any

    """
    Class that is used to represent a GP individual.
    Formally a GP individual can be represented as a tuple consisting of
    the genome and the fitness value.
    """

    def __init__(self, genome_: any, fitness_: any = None):
        self.genome = genome_
        self.fitness = fitness_

@dataclass
class GPLogging:
    best_fitness: float
    best_individual: GPIndividual
    num_evaluation: float

class GPModel(ABC):
    """
    Abstract based class for tiny GP representation modules.
    It describes the minimum requirements for a GP model that is
    integrated in the framework.
    """
    best_individual: GPIndividual
    num_evaluation: float
    population: List[GPIndividual]
    
    def evaluate(self) -> GPIndividual:
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        best = None
        for individual in self.population:
            genome = individual.genome
            if (individual.fitness is None):
                individual.fitness = self.evaluate_individual(genome)
            fitness = individual.fitness

            if self.problem.is_ideal(fitness):
                return individual, True

            if best is None:
                best = individual
                best_fitness = fitness

            if self.problem.is_better(fitness, best_fitness):
                best = individual
                best_fitness = fitness
        
        self.best = best

        return best
        
    @abstractmethod
    def evaluate_individual(self,genome:GPIndividual) -> float:
        """
        Fitness function that evaluates a single individual.
        """
        pass
    
    @abstractmethod
    def evolve(self)  -> Any:
        """
        Main evolution loop that is used to run instances
        of a GP model.
        """
        pass

    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass

    @abstractmethod
    def predict(self) -> Any:
        """
        The respective prediction method is implemented here.
        """
        pass

    @abstractmethod
    def expression(self) -> Any:
        """
        Returns a human-readable solution of a evolved candidate solution.
        Return value can be a string or a list of strings.
        """
        pass

    def report_job(self, job: int, num_evaluations: int, best_fitness: float,
                   silent_evolver: bool, minimalistic_output: bool):
        """
        Report the status of a job after has been executed.

        :param job: job number
        :param num_evaluations: Number of evaluations the run lasted
        :param best_fitness: Best fitness found during the run
        :param silent_evolver: Switch for activating/deactivating the report function
        :param minimalistic_output: Swith for minimalistic output
        :return:
        """
        if not silent_evolver:
            if not minimalistic_output:
                print("Job #" + str(job) + " - Evaluations: " + str(num_evaluations) +
                      " - Best Fitness: " + str(best_fitness))
            else:
                print(str(num_evaluations) + ";" + str(best_fitness))

    def report_generation(self, silent: bool, generation: int, best_fitness: float, report_interval: int):
        """
        Reports the status of a generation.

        :param silent: Switch for activating/deactivating the report function
        :param generation: Generation number
        :param best_fitness: Best fitness found so far
        :param report_interval: Interval after which the generation status is reported
        """
        if not silent and generation % report_interval == 0:
            print("Generation #" + str(generation) + " - Best Fitness: " + str(best_fitness))

@dataclass
class Config(ABC):
    """
    Abstract class for configuration classes.
    """
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPConfig(Config):
    """
    Configuration class for GP models.
    This class contains the common configuration parameters for GP models related to
    execution and output of a run.
    """
    num_jobs: int
    max_generations: int
    stopping_criteria: float
    minimizing_fitness: bool
    ideal_fitness: float
    silent_algorithm: bool
    silent_evolver: bool
    minimalistic_output: bool
    num_outputs: int
    report_interval: int
    max_time: int

@dataclass
class Hyperparameters(ABC):
    """
    Base class for the GP hyperparamters.
    """

    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPHyperparameters(Hyperparameters):
    """
    Hyperparameters class for Genetic Programming models.
    This class is responsible for storing the tunable hyperparameters of the GP model.
    """
    pop_size: int
    max_size: int
    max_depth: int
    mutation_rate: float
    cx_rate: float
    tournament_size: int

@dataclass
class Function():
    """
    Main class used for representing functions, variables or constants.
    It contains information about arity, a string representation for the function,
    and the function itself.
    The method `call` is used to call the function with the given arguments.
    The method `custom` is meant to make the function compatible with sympy, if
    None is passed, it will use the defaults.
    """
    name: str
    arity: int
    function: callable
    custom: callable

    def __init__(self, arity_, name_, function_, custom_=None):
        self.function = function_
        self.name = name_
        self.arity = arity_
        self.custom = custom_

    def __call__(self, *args) -> Any:
        assert (len(args) == self.arity)
        return self.function(*args)


class Var(Function):
    """
    Class for representing variable terminals.
    """
    def __init__(self, index: int = None, name_: str = None):
        self.const = False
        if name_ is None:
            name_ = 'Var'
        Function.__init__(self, 0, name_, lambda: index)


class Const(Function):
    """
    Class for representing constant terminals.
    """
    def __init__(self, value):
        self.const = True
        Function.__init__(self, 0, 'Const', lambda: value)

