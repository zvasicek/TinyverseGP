"""
Benchmark representation module for symbolic regression.
"""

import random
from src.benchmark.benchmark import Benchmark


class SRBenchmark(Benchmark):
    """
    Represents a symbolic regression benchmark that is based on a uniformly sampled dataset.
    """

    def dataset_uniform(
        self, a: int, b: int, n: int, dimension: int, benchmark: str
    ) -> tuple:
        """
        Samples a data net of n datapoints drawn from uniform distribution in the
        closed interval [a, b]

        :param a: minimum
        :param b: maximum
        :param n: number of samples
        :param dimension: dimension of the objective function
        :param benchmark: benchmark function
        :return: samples and corresponding actual function values
        """
        sample = []
        point = []
        for _ in range(n):
            point.clear()
            for _ in range(dimension):
                point.append(random.uniform(a, b))
            sample.append(point.copy())
        values = [self.objective(benchmark, point) for point in sample]
        return sample, values

    def generate(self, benchmark: str) -> tuple:
        """
        Generates the dataset for a selected benchmark function.

        :param benchmark: selected benchmark function
        :return: generated dataset
        """
        match benchmark:
            case "KOZA1":
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)
            case "KOZA2":
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)
            case "KOZA3":
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)

    def objective(self, benchmark: str, args: list) -> float:
        """
        Calculates the actual/objective value of the benchmark function.

        :param benchmark: Selected benchmark function
        :param args: list of arguments

        :return: objective function value
        """
        match benchmark:
            case "KOZA1":
                return pow(args[0], 4) + pow(args[0], 3) + pow(args[0], 2) + args[0]
            case "KOZA2":
                return pow(args[0], 5) - 2 * pow(args[0], 3) + args[0]
            case "KOZA3":
                return pow(args[0], 6) - 2 * pow(args[0], 4) + pow(args[0], 2)
