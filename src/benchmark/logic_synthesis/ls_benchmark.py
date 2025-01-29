from src.benchmark.benchmark import Benchmark

class LSBenchmark(Benchmark):
    """
    Class for representing logic synthesis benchmarks. Receives
    an generator function that is used to create the dataset.
    """
    def __init__(self, generator_ : callable, args_: list):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        """
        Calls the generator function and triggers the generations of the dataset.
        """
        self.dataset = self.generator(*self.args)