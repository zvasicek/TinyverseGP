from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class Benchmark(ABC):
    """
    Abstract base class for the representation of benchmarks.
    """

    @abstractmethod
    def generate(self, args: any):
        """
        Generator function that creates the benchmark with respect to the
        problem type.
        """
        pass