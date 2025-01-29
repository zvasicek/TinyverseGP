from src.benchmark.benchmark import Benchmark

class PSBenchmark(Benchmark):
    """
    Represents a program synthesis benchmark that is
    represented with examples/counterexamples
    """
    generator: callable
    args: list
    dataset: list

    def __init__(self, generator_: callable, args_: list):
        self.generator = generator_
        self.args = args_
        self.generate()

    def generate(self):
        self.dataset = self.generator(*self.args)

    def generate_counterexamples(self, examples: list, n: int):
        """
        Calculates the counter examples based on the set of (positive) examples
        :param n: Number of counter examples to be generated
        """
        counterexamples = [i for i in range(n) if n not in examples]
        return counterexamples

    def generate_dataset(self, examples: list, n: int):
        """
        Generate a dataset of n examples
        :param n: Number of examples
        """
        dataset = []
        for i in range(n):
            if i in examples:
                dataset.append((i, 1))
            else:
                dataset.append((i, 0))
        return dataset
