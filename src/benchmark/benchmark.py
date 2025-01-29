from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class Benchmark(ABC):
    """
    Abstract base class for benchmarks.
    """

    @abstractmethod
    def generate(self, benchmark: str):
        """
        Generator function that creates the benchmark with respect to the
        problem type.
        """
        pass

    @abstractmethod
    def objective(self, benchmark: str, args: list):
        """
        Calculates actual values of an objective function that is going to
        be approximated
        """
        pass