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

    @abstractmethod
    def report_job(self):
        pass

    @abstractmethod
    def report_generation(self):
        pass

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


