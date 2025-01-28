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

    def generate_counterexamples(examples, n):
        counterexamples = [i for i in range(n) if n not in examples]
        return counterexamples

    def generate_dataset(examples, n):
        dataset = []
        for i in range(n):
            if i in examples:
                dataset.append((i, 1))
            else:
                dataset.append((i, 0))
        return dataset
