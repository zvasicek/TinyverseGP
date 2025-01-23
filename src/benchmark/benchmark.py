from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class Benchmark(ABC):
    @abstractmethod
    def generate(self, benchmark: str):
        pass

    @abstractmethod
    def objective(self, benchmark: str, args: list):
        pass