from src.benchmark.benchmark import Benchmark

class LSBenchmark(Benchmark):

    def __init__(self, generator_, args_):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        self.dataset = self.generator(*self.args)