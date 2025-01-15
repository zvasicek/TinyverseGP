# GPBench TinyCGP 0.1
# Roman Kalkreuth (roman.kalkreuth@rwth-aachen.de)
# RWTH Aachen University (Germany)

__author__ = 'Roman Kalkreuth'
__version__ = '0.1'
__email__ = 'roman.kalkreuth@rwth-aachen.de'

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import math
import random
import operator
import copy

MU = 1
LAMBDA = 1
POPULATION_SIZE = MU + LAMBDA
NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_FUNCTION_NODES = 10
MAX_ARITY = 2
NUM_NODES = NUM_INPUTS + NUM_FUNCTION_NODES + NUM_OUTPUTS
NUM_GENES = ((MAX_ARITY + 1) * NUM_FUNCTION_NODES) + NUM_OUTPUTS
LEVELS_BACK = 1
NUM_FUNCTIONS = 4
MUTATION_RATE = 0.1
GENERATIONS = 100
STRICT_SELECTION = True
IDEAL = 0.01
MINIMIZING = True
SEED = 42

def pdiv(x, y):
    return x / y if y > 0 else 1.0

def hamming_distance(x:dict, y:dict) -> int:
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            dist += 1

def euclidean_distance(x:dict, y:dict) -> float:
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += math.pow(xi - yi, 2)
    return math.sqrt(dist)


@dataclass
class Terminal(ABC):
    index: int
    type: int

@dataclass
class Functions():
    functions: list[callable]
    names: list[str]

    def __init__(self, functions_, names_):
        self.functions = functions_

    def call(self, index: int, args: list) -> float:
        if len(args) == 1:
            return self.functions[index](args[0])
        else:
            return self.functions[index](args[0], args[1])

    def name(self, index: int) -> str:
        return self.names[index]

@dataclass
class Problem():
    data: list
    actual: list

    def __init__(self, data_: list, actual_: list, loss_: callable(),
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
                return pow(args[0],4) + pow(args[0],3) + pow(args[0],2) + args[0]
            case 'KOZA2':
                return pow(args[0],5) - 2*pow(args[0],3) + args[0]
            case 'KOZA3':
                return pow(args[0],5) - 2 * pow(args[0],4) + pow(args[0],2)


def check_config():
    if LEVELS_BACK > NUM_FUNCTION_NODES:
        raise ValueError('LEVELS_BACK > NUM_FUNCTION_NODES')


class TinyCGP:
    class GeneType(Enum):
        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        VARIABLE = 0
        CONSTANT = 1

    def __init__(self, problem_: object, functions_: Functions, terminals_: list):
        self.population = []
        self.functions = functions_
        self.inputs = dict()
        self.problem = problem_
        self.init_inputs(terminals_)
        self.init_population()

    def init_population(self):
        for _ in range(POPULATION_SIZE):
            individual = self.init_individual()
            self.population.append(individual)

    def init_individual(self):
        genome = self.init_genome()
        fitness = 0.0
        return [genome, fitness]

    def init_inputs(self, terminals_: list):
        for index, terminal in enumerate(terminals_):
            if isinstance(terminal, str):
                self.inputs[index] = (terminal, self.TerminalType.VARIABLE)
            else:
                self.inputs[index] = (terminal, self.TerminalType.CONSTANT)

    def init_genome(self) -> list[int]:
        genome = []
        for count in range(NUM_GENES):
            gene = self.init_gene(count)
            genome.append(gene)
        return genome

    def init_gene(self, position: int) -> int:
        gene_type = self.phenotype(position)
        if gene_type == self.GeneType.CONNECTION:
            node_num = self.node_number(position)
            if node_num <= LEVELS_BACK:
                return random.randint(0, node_num - 1)
            else:
                return random.randint(node_num - LEVELS_BACK, node_num - 1)
        elif gene_type == self.GeneType.FUNCTION:
            return random.randint(0, NUM_FUNCTIONS - 1)
        else:
            return random.randint(0, NUM_FUNCTION_NODES + NUM_INPUTS - 1)

    def phenotype(self, position: int) -> GeneType:
        if position >= NUM_FUNCTION_NODES * (MAX_ARITY + 1):
            return self.GeneType.OUTPUT
        else:
            return self.GeneType.FUNCTION if position % (MAX_ARITY + 1) == 0 else self.GeneType.CONNECTION

    def input_value(self, index: int) -> any:
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        return str(self.input_value(index))

    def input_type(self, index: int) -> TerminalType:
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        return math.floor(position / (MAX_ARITY + 1)) + NUM_INPUTS

    def node_position(self, node_num: int) -> int:
        return (node_num - NUM_INPUTS) * (MAX_ARITY + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> int:
        position = self.node_position(node_num)
        inputs = []
        for count in range(MAX_ARITY):
            inputs.append(genome[position + count + 1])
        return inputs

    def outputs(self, genome: list[int]) -> list[int]:
        outputs = []
        for output_pos in range(NUM_GENES - 1, NUM_GENES - NUM_OUTPUTS - 1, -1):
            output = genome[output_pos]
            outputs.append(output)
        return outputs

    def max_gene(self, position: int):
        if self.phenotype(position) == self.GeneType.OUTPUT:
            return NUM_INPUTS + NUM_FUNCTIONS - 1
        elif self.phenotype(position) == self.GeneType.CONNECTION:
            return NUM_FUNCTIONS - 1
        else:
            return self.node_number(position) - 1

    def fitness(self, individual: list[list[int], float]):
        return individual[1]

    def evaluate(self) -> float:
        best = None
        for individual in self.population:
            genome = individual[0]
            fitness = individual[1] = self.evaluate_genome(genome)

            if best is None:
                best = fitness

            if self.problem.is_ideal(fitness):
                return individual
            if self.problem.is_better(fitness, best):
                best = fitness
        return best
                
    def evaluate_genome(self, genome: list[int]) -> float:
        predictions = []
        paths = self.decode(genome)
        for data_point in self.problem.data:
            prediction = self.predict(genome, data_point, paths)
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

    def predict(self, genome: list[int], data_point: list, paths = None) -> list:
        node_map = dict()
        prediction = []

        if paths is None:
            paths = self.decode(genome)

        for path in paths:
            cost = 0.0
            for node_num in path:
                if node_num not in node_map.keys():
                    if node_num < NUM_INPUTS:
                        if self.input_type(node_num) == self.TerminalType.VARIABLE:
                            node_map[node_num] = data_point[node_num]
                        else:
                            node_map[node_num] = self.input_value(node_num)
                    else:
                        arguments = []
                        connections = self.node_connections(node_num, genome)
                        for count, connection in enumerate(connections):
                            argument = node_map[connection]
                            arguments.append(argument)
                        node_map[node_num] = self.evaluate_node(node_num, genome, arguments)
                cost = node_map[node_num]
            prediction.append(cost)
        return prediction

    def active_nodes(self, genome: list[int]):
        nodes = dict()

        for position in range(NUM_GENES - 1, NUM_GENES - NUM_OUTPUTS - 1, -1):
            node_num = genome[position]
            if node_num >= NUM_INPUTS:
                nodes[node_num] = True

        for position in range(NUM_GENES - NUM_OUTPUTS - 1, 0, -1):
            node_num = self.node_number(position)
            if (node_num in nodes.keys()
                    and self.phenotype(position) == self.GeneType.CONNECTION):
                gene = genome[position]
                if gene >= NUM_INPUTS:
                    nodes[gene] = True
        return sorted(nodes.keys())

    def decode(self, genome: list[int]):
        paths = []
        nodes = dict()
        for output_pos in range(NUM_GENES - 1, NUM_GENES - NUM_OUTPUTS - 1, -1):
            nodes.clear()
            node_num = genome[output_pos]
            nodes[node_num] = True

            for gene_pos in range(NUM_GENES - NUM_OUTPUTS - 1, 0, -1):
                if (self.node_number(gene_pos) in nodes.keys()
                        and self.phenotype(gene_pos) == self.GeneType.CONNECTION):
                    gene = genome[gene_pos]
                    nodes[gene] = True
            path = sorted(nodes.keys())
            paths.append(path)
        return paths

    def evolve(self):
        for generation in range(GENERATIONS):
            self.breed()
            best = self.evaluate()
            print("Generation #" + str(generation) + " -> Best Fitness: " + str(best))

    def breed(self):
        parent = self.selection()
        self.population.clear()
        self.population.append(parent)
        for _ in range(LAMBDA):
            offspring = copy.deepcopy(parent)
            self.mutation(offspring[0])
            self.population.append(offspring)

    def selection(self) -> list:
        sorted_pop = sorted(self.population, key=lambda ind: ind[1])
        count = 0
        if not STRICT_SELECTION:
            best_fitness = sorted_pop[0]
            for individual in sorted_pop:
                if individual[1] != best_fitness:
                    break
                else:
                    count += 1
        parent = random.randint(0, count)
        return sorted_pop[parent]

    def mutation(self, genome: list[int]):
        for index, gene in enumerate(genome):
            if random.random() < MUTATION_RATE:
                genome[index] = self.init_gene(index)

    def expression(self, genome: list[int]) -> list[str]:

        def generate_expr_map(genome: list[int], active_nodes=None):
            if active_nodes is None:
                active_nodes = self.active_nodes(genome)
            expr_map = dict()
            for node_num in active_nodes:
                args = self.node_connections(node_num, genome)
                function = self.node_function(node_num, genome)
                func_name = self.functions.name(function)
                node_expr = func_name + "("
                for index, argument in enumerate(args):
                    if argument in expr_map:
                        arg_expr = expr_map[argument]
                    else:
                        arg_expr = self.input_name(argument)
                    node_expr += arg_expr
                    if index < MAX_ARITY - 1:
                        node_expr += ", "
                node_expr += ")"
                expr_map[node_num] = node_expr
            return expr_map

        expr_map = generate_expr_map(genome)
        expressions = []

        outputs = self.outputs(genome)

        for output in outputs:
            if output < NUM_INPUTS:
                expression = self.input_name(output)
            else:
                expression = expr_map[output]
            expressions.append(expression)

        return expressions

    def print_population(self):
        for individual in self.population:
            print(individual)

random.seed(SEED)
functions = Functions([operator.add, operator.sub, operator.mul, pdiv],
                      ["ADD", "SUB", "MUL", "DIV"])
terminals = ['X', 1]
loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate('KOZA1')
problem = Problem(data, actual, loss, IDEAL, MINIMIZING)
cgp = TinyCGP(problem, functions, terminals)
cgp.evolve()
