# GPBench TinyGP 0.1
# Fabricio Olivetti de Franca
# Universidade Federal do ABC 
# Based on tinyGP by Moshe Sipper - https://github.com/moshesipper/tiny_gp/

__author__ = 'Fabricio Olivetti de Franca'
__version__ = '0.1'
__email__ = 'folivetti@ufabc.edu.br'

import math
import random
import operator
import copy

from abc import ABC
from abc import abstractmethod
from typing import List, Any
from dataclasses import dataclass

POP_SIZE = 10
MAX_SIZE = 10
MAX_DEPTH = 5
MUTATION_RATE = 0.1
CX_RATE = 0.9
TOURNAMENT_SIZE = 2
GENERATIONS = 100
JOBS = 10
IDEAL = 0.01
MINIMIZING = True
SEED = 42


def pdiv(x, y):
    return x / y if y > 0 else 1.0


def hamming_distance(x: dict, y: dict) -> int:
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            dist += 1


def euclidean_distance(x: dict, y: dict) -> float:
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += math.pow(xi - yi, 2)
    return math.sqrt(dist)


@dataclass
class Config(ABC):
    def dictionary(self) -> dict:
        return self.__dict__


@dataclass
class GPConfig(Config):
    num_jobs: int
    max_generations: int
    stopping_criteria: float
    minimizing_fitness: bool
    ideal_fitness: float
    silent_algorithm: bool
    silent_evolver: bool
    minimalistic_output: bool


class Hyperparameters(ABC):
    def dictionary(self) -> dict:
        return self.__dict__


@dataclass
class GPHyperparameters(Hyperparameters):
    pop_size: int
    max_size: int
    max_depth: int
    mutation_rate: float
    cx_rate: float
    tournament_size: int


@dataclass
class Function():
    # NOTE: using Function instead of Functions will help to make easier to expand 
    name: str
    arity: int
    function: callable

    def __init__(self, arity_, name_, function_):
        self.function = function_
        self.name = name_
        self.arity = arity_

    def call(self, args: list) -> Any:
        assert (len(args) == self.arity)
        return self.function(*args)


class Var(Function):
    def __init__(self, index):
        self.const = False
        Function.__init__(self, 0, 'Var', lambda: index)


class Const(Function):
    def __init__(self, value):
        self.const = True
        Function.__init__(self, 0, 'Const', lambda: value)

    # SR Functions


Add = Function(2, 'Add', operator.add)
Sub = Function(2, 'Sub', operator.sub)
Mul = Function(2, 'Mul', operator.mul)
Div = Function(2, 'Div', pdiv)


@dataclass
class Problem():
    data: list
    actual: list

    def __init__(self, data_: list, actual_: list, loss_: callable,
                 ideal_: float, minimizing_: bool):
        self.data = data_
        self.actual = actual_
        self.loss = loss_
        self.ideal = ideal_
        self.minimizing = minimizing_

    def evaluate(self, prediction: list) -> float:
        return self.loss(self.actual, prediction)

    def is_ideal(self, fitness: float) -> bool:
        return fitness == self.ideal

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        return fitness1 < fitness2 if self.minimizing \
            else fitness1 > fitness2


@dataclass
class Benchmark(ABC):
    @abstractmethod
    def generate(self, benchmark: str):
        pass

    @abstractmethod
    def objective(self, benchmark: str, args: list):
        pass


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
                return pow(args[0], 5) - 2 * pow(args[0], 4) + pow(args[0], 2)


class Node:
    def __init__(self, function: Any, children: List[Any]):
        self.function = function
        self.children = children


def node_size(node: Node) -> int:
    if len(node.children) == 0:
        return 1
    return 1 + sum([node_size(child) for child in node.children])


class TinyGP:
    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: list[Function]

    def __init__(self, problem_: object, functions_: list[Function], config: Config, hyperparameters: Hyperparameters):
        # NOTE: having init_population as a function could allow people to call it again and double the pop size 
        self.functions = [f for f in functions_ if f.arity > 0]
        self.terminals = [f for f in functions_ if f.arity == 0]
        self.problem = problem_
        self.hyperparameters = hyperparameters
        self.config = config
        self.population = [[genome, 0.0] for genome in self.init_ramped_half_half(POP_SIZE, 1, MAX_DEPTH, MAX_SIZE)]
        self.evaluate()

    def init_individual(self):
        genome = self.init_genome()
        fitness = 0.0
        return [genome, fitness]

    def tree_random_full(self, max_depth: int, size: int) -> Node:
        """
        returns a random tree with `max_depth` and using functions
        that sample random terminal and nonterminal nodes.
        """
        if max_depth == 0 or size < 2:
            return Node(random.choice(self.terminals), [])
        n = random.choice(self.functions)
        children = [self.tree_random_full(max_depth - 1, size // n.arity - 1) for _ in range(n.arity)]
        return Node(n, children)

    def tree_random_grow(self, min_depth: int, max_depth: int, size: int) -> Node:
        """
        returns a random tree with `max_depth` and using functions
        that sample random terminal and nonterminal nodes.
        """
        if max_depth <= 1 or size < 2:
            return Node(random.choice(self.terminals), [])
        if min_depth <= 0 and random.random() < 0.5:
            return Node(random.choice(self.terminals), [])
        else:
            n = random.choice(self.functions)
            size = size - n.arity
            children = []
            for _ in range(n.arity):
                child = self.tree_random_grow(min_depth - 1, max_depth - 1, size)
                size -= node_size(child)
                children.append(child)
        return Node(n, children)

    def init_ramped_half_half(self, num_pop: int, min_depth: int, max_depth: int, max_size: int):
        pop = []
        for md in range(min_depth, max_depth + 1):
            grow = True
            for _ in range(int(num_pop / (max_depth - 3 + 1))):
                if grow:
                    tree = self.tree_random_grow(min_depth, max_depth, max_size)
                else:
                    tree = self.tree_random_full(max_depth, max_size)
                pop.append(tree)
                grow = not grow
        return pop

    def fitness(self, individual: list[list[int], float]):
        return individual[1]

    def evaluate(self) -> float:
        best = None
        for ix, individual in enumerate(self.population):
            genome = individual[0]
            fitness = self.evaluate_genome(genome)
            self.population[ix][1] = fitness

            if best is None:
                best = fitness

            if self.problem.is_ideal(fitness):
                return fitness
            if self.problem.is_better(fitness, best):
                best = fitness
        return best

    def evaluate_genome(self, genome: Node) -> float:
        predictions = []
        for data_point in self.problem.data:
            prediction = self.predict(genome, data_point)
            predictions.append(prediction)
        return self.cost(predictions)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        function = self.node_function(node_num, genome)
        return self.functions.call(function, args)

    def cost(self, predictions: list) -> float:
        cost = 0.0
        for index, _ in enumerate(predictions[0]):
            cost += self.problem.evaluate([prediction[index] for prediction in predictions])
        return cost

    def predict(self, genome: Node, data_point: list) -> list:
        prediction = []

        def eval_node(node: Node):
            if node.function.arity == 0:
                # NOTE: this assumes that we will first have the list of variables in order in the list of terminals 
                if node.function.const:
                    return node.function.call([])
                else:
                    return data_point[node.function.call([])]
            else:
                args = [eval_node(child) for child in node.children]
                return node.function.call(args)

        # TODO: deal with multiple outputs by using multiple trees 
        prediction.append(eval_node(genome))
        return prediction

    def breed(self):
        parents = [[self.selection(), self.selection()] for _ in range(self.hyperparameters.pop_size)]
        self.population = [self.perturb(*parent) for parent in parents]

    def perturb(self, parent1: Node, parent2: Node) -> list:
        genome = self.crossover(parent1, parent2) if random.random() <= self.hyperparameters.cx_rate else parent1
        genome = self.mutation(genome, self.hyperparameters.max_depth,
                               self.hyperparameters.max_size) if random.random() <= self.hyperparameters.mutation_rate else genome
        return [genome, self.evaluate_genome(genome)]

    def selection(self) -> Node:
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        if problem.minimizing:
            return min(parents, key=lambda ind: ind[1])[0]
        else:
            return max(parents, key=lambda ind: ind[1])[0]

    def crossover(self, p1: Node, p2: Node) -> Node:
        def pick_from(n: Node, ix: int) -> Node:
            if ix == 0:
                return n
            tryout = None
            ix = ix - 1
            for iy in range(n.function.arity):
                tryout = pick_from(n.children[iy], ix)
                ix = ix - node_size(n.children[iy])
                if tryout is not None:
                    break
            return tryout

        def assemble(n1: Node, n2: Node, ix: int) -> Node:
            if ix == 0:
                return n2
            new_node = copy.deepcopy(n1)
            children = []
            ix = ix - 1
            for child in new_node.children:
                children.append(assemble(child, n2, ix) if ix > 0 else child)
                ix = ix - node_size(child)
            return Node(n1.function, children)

        piece2 = pick_from(p2, random.choice(range(node_size(p2))))
        return assemble(p1, piece2, random.choice(range(node_size(p1))))

    def mutation(self, n: Node, max_depth: int, size: int):
        n_nodes = node_size(n)
        ix = random.choice(range(n_nodes))

        def traverse(n: Node, iy: int, maxD: int, sz: int) -> Node:
            if iy == 0:
                return self.tree_random_grow(1, maxD, sz)
            if iy < 0:
                return copy.deepcopy(n)
            children = []
            iy = iy - 1
            maxD = maxD - 1
            sz = sz - 1
            for child in n.children:
                children.append(traverse(child, iy, maxD, sz))
                iy = iy - node_size(children[-1])
                maxD = maxD - 1
                sz = sz - node_size(children[-1])
            return Node(n.function, children)

        return traverse(n, ix, max_depth, size)

    def expression(self, genome: Node) -> list[str]:

        def print_node(node: Node):
            if len(node.children) == 0:
                return node.function.name + "(" + str(node.function.call([])) + ")"
            else:
                args = [print_node(child) for child in node.children]
                return node.function.name + "(" + ", ".join(args) + ")"

        # TODO: multi-tree
        return [print_node(genome)]

    def print_population(self):
        for individual in self.population:
            self.print_individual(individual)

    def print_individual(self, individual):
        print("Genome: " + ";".join(self.expression(individual[0])) + " : Fitness: " + str(individual[1]))

    def evolve(self):
        # NOTE: is this supposed as a preparation for multithreading?
        for job in range(JOBS):
            best_fitness = None
            for generation in range(GENERATIONS):
                self.breed()
                best_fitness = self.evaluate()
                print("Generation #" + str(generation) + " -> Best Fitness: " + str(best_fitness))
            self.print_individual(self.population[-1])
            print("Job #" + str(job) + " -> Best Fitness: " + str(best_fitness))


random.seed(SEED)
functions = [Add, Sub, Mul, Div, Var(0), Const(1)]
loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate('KOZA1')
problem = Problem(data, actual, loss, IDEAL, MINIMIZING)
hp = GPHyperparameters(pop_size=POP_SIZE, max_size=MAX_SIZE, max_depth=MAX_DEPTH, cx_rate=CX_RATE,
                       mutation_rate=MUTATION_RATE, tournament_size=TOURNAMENT_SIZE)
config = GPConfig(
    num_jobs=2,
    max_generations=100,
    stopping_criteria=0.01,
    minimizing_fitness=True,
    ideal_fitness=0.01,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True
)

gp = TinyGP(problem, functions, config, hp)
gp.evolve()
