from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any

class GPModel(ABC):
    '''
    Abstract class for Genetic Programming models.
    It describes the minimum requirements for a GP model.
    '''
    best_fitness: float

    def fitness(self, individual):
        '''
        Fitness function of a single individual.
        '''
        return individual[1]

    @abstractmethod
    def evolve(self):
        '''
        Evolve the population.
        '''
        pass

    def selection(self):
        '''
        Selection of individuals for recombination and perturbation.
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        Predict the output of the best individual.
        '''
        pass

    @abstractmethod
    def expression(self):
        '''
        Return the expression of the best individual.
        '''
        pass

    def report_job(self, job: int, num_evaluations: int, best_fitness: float,
                   silent_evolver: bool, minimalistic_output: bool):
        if not silent_evolver:
            if not minimalistic_output:
                print("Job #" + str(job) + " - Evaluations: " + str(num_evaluations) +
                      " - Best Fitness: " + str(best_fitness))
            else:
                print(str(num_evaluations) + ";" + str(best_fitness))

    def report_generation(self, silent: bool, generation: int, best_fitness: float, report_interval: int):
        if not silent and generation % report_interval == 0:
            print("Generation #" + str(generation) + " - Best Fitness: " + str(best_fitness))

@dataclass
class Config(ABC):
    '''
    Abstract class for configuration classes.
    '''
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPConfig(Config):
    '''
    Configuration class for Genetic Programming models.
    This class contains the common configuration parameters for GP models related to 
    execution and stopping criteria.
    '''
    num_jobs: int
    max_generations: int
    stopping_criteria: float
    minimizing_fitness: bool
    ideal_fitness: float
    silent_algorithm: bool
    silent_evolver: bool
    minimalistic_output: bool
    num_outputs: int
    report_interval:int
    max_time:int

@dataclass
class Hyperparameters(ABC):
    '''
    Abstract class for hyperparameters classes.
    '''
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPHyperparameters(Hyperparameters):
    '''
    Hyperparameters class for Genetic Programming models.
    This class is responsible for storing the tunable hyperparameters of the GP model.
    '''
    pop_size: int
    max_size: int
    max_depth: int
    mutation_rate: float
    cx_rate: float
    tournament_size: int

@dataclass
class Function():
    '''
    Abstract class for function classes.
    It contains information about arity, a string representation for the function, 
    and the function itself.
    The method `call` is used to call the function with the given arguments.
    The method `custom` is meant to make the function compatible with sympy, if
    None is passed, it will use the defaults.
    '''
    name: str
    arity: int
    function: callable
    custom: callable

    def __init__(self, arity_, name_, function_, custom_=None):
        self.function = function_
        self.name = name_
        self.arity = arity_
        self.custom = custom_

    def call(self, args: list) -> Any:
        assert (len(args) == self.arity)
        return self.function(*args)


class Var(Function):
    '''
    Variable function class.
    '''
    def __init__(self, index:int , name_:str = None):
        self.const = False
        if name_ is None:
            name_ = 'Var'
        Function.__init__(self, 0, name_, lambda: index)


class Const(Function):
    '''
    Constant function class.
    '''
    def __init__(self, value):
        self.const = True
        Function.__init__(self, 0, 'Const', lambda: value)

