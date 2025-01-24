from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any

class GPModel(ABC):
    best_fitness: float

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def evolve(self):
        pass

    def selection(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def expression(self):
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
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPConfig(Config):
    num_jobs: int
    max_generations: int
    stopping_criteria: float
    minimizing_fitness: bool
    ideal_fitness: float
    silent_algorithm: bool
    silent_evolver: bool
    minimalistic_output: bool
    report_interval:int

@dataclass
class Hyperparameters(ABC):
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class GPHyperparameters(Hyperparameters):
    pop_size: int
    max_size: int
    max_depth: int
    mutation_rate: float
    cx_rate: float
    tournament_size: int

@dataclass
class Function():
    name: str
    arity: int
    function: callable

    def __init__(self, arity_, name_, function_):
        self.function = function_
        self.name = name_
        self.arity = arity_

    def call(self, args: list) -> Any:
        assert (len(args) == self.arity)
        return self.function(*args)


class Var(Function):
    def __init__(self, index:int , name_:str = None):
        self.const = False
        if name_ is None:
            name_ = 'Var'
        Function.__init__(self, 0, name_, lambda: index)


class Const(Function):
    def __init__(self, value):
        self.const = True
        Function.__init__(self, 0, 'Const', lambda: value)

