import random
from src.benchmark.benchmark import Benchmark

class SRBenchmark(Benchmark):
    def dataset_uniform(self, a: int, b: int, n: int, dimension: int, benchmark: str) -> tuple:
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
        match benchmark:
            case 'KOZA1':
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)
            case 'KOZA2':
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)
            case 'KOZA3':
                return self.dataset_uniform(-1, 1, 20, 1, benchmark)

    def objective(self, benchmark: str, args: list) -> float:
        match benchmark:
            case 'KOZA1':
                return pow(args[0], 4) + pow(args[0], 3) + pow(args[0], 2) + args[0]
            case 'KOZA2':
                return pow(args[0], 5) - 2 * pow(args[0], 3) + args[0]
            case 'KOZA3':
                return pow(args[0], 6) - 2 * pow(args[0], 4) + pow(args[0], 2)