from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

class GPBase(ABC):
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

    def report_generation(self, silent: bool, generation: int, best_fitness: float):
        if not silent:
            print("Generation #" + str(generation) + " - Best Fitness: " + str(best_fitness))

@dataclass
class Config(ABC):
    def dictionary(self) -> dict:
        return self.__dict__

class Hyperparameters(ABC):
    def dictionary(self) -> dict:
        return self.__dict__

@dataclass
class Benchmark(ABC):
    @abstractmethod
    def generate(self, benchmark: str):
        pass

    @abstractmethod
    def objective(self, benchmark: str, args: list):
        pass

@dataclass
class Problem():
    data: list
    actual: list

    def evaluate(self, prediction: list) -> float:
        pass


