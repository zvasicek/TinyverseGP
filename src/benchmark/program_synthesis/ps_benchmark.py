from src.benchmark import benchmark

class PSBenchmark(benchmark):
    generator: callable
    args: list
    dataset: list

    def __init__(self, generator_, args_):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        self.dataset = self.generator(*self.args)
