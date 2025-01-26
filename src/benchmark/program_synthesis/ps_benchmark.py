from src.benchmark.benchmark import Benchmark

class PSBenchmark(Benchmark):
    generator: callable
    args: list
    dataset: list

    def __init__(self, generator_, args_):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        self.dataset = self.generator(*self.args)
