# GPBench TinyCGP 0.1
# Roman Kalkreuth (roman.kalkreuth@rwth-aachen.de)
# RWTH Aachen University (Germany)

__author__ = 'Roman Kalkreuth'
__version__ = '0.1'
__email__ = 'roman.kalkreuth@rwth-aachen.de'

import math
import random
import copy

from dataclasses import dataclass
from enum import Enum


from tiny_gp import (TinyGP, Problem, GPConfig, Hyperparameters, SRBenchmark, euclidean_distance,
                     Add, Sub, Mul, Div, Var, Const)
from src.benchmark.policy_search.policy_evaluation import GPAgent
from src.gp.problem import PolicySearchProblem
from src.gp.problem import BlackBoxProblem

import gymnasium as gym

MU = 1
LAMBDA = 1
POPULATION_SIZE = MU + LAMBDA
NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_FUNCTION_NODES = 10
MAX_ARITY = 2
NUM_NODES = NUM_INPUTS + NUM_FUNCTION_NODES + NUM_OUTPUTS
NUM_GENES = ((MAX_ARITY + 1) * NUM_FUNCTION_NODES) + NUM_OUTPUTS
LEVELS_BACK = 10
NUM_FUNCTIONS = 4
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100
NUM_JOBS = 10
STRICT_SELECTION = True
IDEAL_FITNESS = 0.01
MINIMIZING_FITNESS = True
SEED = 42

SILENT_ALGORITHM = False
SILENT_EVOLVER = False
MINIMALISTIC_OUTPUT = False


@dataclass
class CGPHyperparameters(Hyperparameters):
    mu: int
    lmbda: int
    population_size: int
    levels_back: int
    mutation_rate: float
    strict_selection: bool

@dataclass
class CGPConfig(GPConfig):
    num_inputs: int
    num_outputs: int
    num_function_nodes: int
    num_genes: int
    num_functions: int
    max_arity: int

def check_config():
    if LEVELS_BACK > NUM_FUNCTION_NODES:
        raise ValueError('LEVELS_BACK > NUM_FUNCTION_NODES')

class TinyCGP(TinyGP):

    class GeneType(Enum):
        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        VARIABLE = 0
        CONSTANT = 1

    def __init__(self, problem_: Problem, functions_: list, terminals_: list,
                 config_: GPConfig, hyperparameters_: Hyperparameters, evaluator_ = None):
        #TinyGP.__init__(problem_, functions_, config_, hyperparameters_)
        self.population = []
        self.functions = functions_
        self.terminals = terminals_
        self.problem = problem_
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.inputs = dict()
        self.init_inputs(terminals_)
        self.init_population()
        if evaluator_ is not None:
            self.evaluator = evaluator_
        else:
            self.evaluator = self.evaluate_observations

    def init_population(self):
        for _ in range(self.hyperparameters.population_size):
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
        for count in range(self.config.num_genes):
            gene = self.init_gene(count)
            genome.append(gene)
        return genome

    def init_gene(self, position: int) -> int:
        gene_type = self.phenotype(position)
        levels_back =  self.hyperparameters.levels_back
        if gene_type == self.GeneType.CONNECTION:
            node_num = self.node_number(position)
            if node_num <= levels_back:
                return random.randint(0, node_num - 1)
            else:
                return random.randint(node_num - levels_back, node_num - 1)
        elif gene_type == self.GeneType.FUNCTION:
            return random.randint(0, self.config.num_functions - 1)
        else:
            return random.randint(0, self.config.num_function_nodes + self.config.num_inputs - 1)

    def phenotype(self, position: int) -> GeneType:
        if position >= self.config.num_function_nodes * (self.config.max_arity + 1):
            return self.GeneType.OUTPUT
        else:
            return self.GeneType.FUNCTION if position % (self.config.max_arity + 1) == 0 \
                else self.GeneType.CONNECTION

    def input_value(self, index: int) -> any:
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        return str(self.input_value(index))

    def input_type(self, index: int) -> TerminalType:
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        return math.floor(position / (self.config.max_arity + 1)) + self.config.num_inputs

    def node_position(self, node_num: int) -> int:
        return (node_num - self.config.num_inputs) * (self.config.max_arity + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> int:
        position = self.node_position(node_num)
        inputs = []
        for count in range(self.config.max_arity):
            inputs.append(genome[position + count + 1])
        return inputs

    def outputs(self, genome: list[int]) -> list[int]:
        outputs = []
        for output_pos in range(self.config.num_genes - 1,
                                self.config.num_genes - self.config.num_outputs - 1, -1):
            output = genome[output_pos]
            outputs.append(output)
        return outputs

    def max_gene(self, position: int):
        if self.phenotype(position) == self.GeneType.OUTPUT:
            return self.config.num_inputs + self.config.num_functions - 1
        elif self.phenotype(position) == self.GeneType.CONNECTION:
            return self.config.num_functions - 1
        else:
            return self.node_number(position) - 1

    def fitness(self, individual: list[list[int], float]):
        return individual[1]

    def evaluate(self) -> float:
        best = None
        for individual in self.population:
            genome = individual[0]
            fitness = individual[1] = self.evaluate_individual(genome)

            if best is None:
                best = fitness

            if self.problem.is_ideal(fitness):
                return individual
            if self.problem.is_better(fitness, best):
                best = fitness
        return best

    def evaluate_individual(self, genome:list[int], evaluator) -> float:
        return evaluator(genome)

    def evaluate_observation(self, genome: list[int], observation):
        paths = self.decode(genome)
        return self.predict(genome, observation, paths)

    def evaluate_observations(self, genome: list[int]) -> float:
        predictions = []
        paths = self.decode(genome)
        for observation in self.problem.data:
            prediction = self.predict(genome, observation, paths)
            predictions.append(prediction)
        return self.cost(predictions)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        function = self.node_function(node_num, genome)
        return self.functions[function].call(args)

    def cost(self, predictions: list) -> float:
        cost = 0.0
        for index, _ in enumerate(predictions[0]):
            cost += self.problem.evaluate([prediction[index] for prediction in predictions])
        return cost

    def predict(self, genome: list[int], data_point: list, paths=None) -> list:
        node_map = dict()
        prediction = []

        if paths is None:
            paths = self.decode(genome)

        for path in paths:
            cost = 0.0
            for node_num in path:
                if node_num not in node_map.keys():
                    if node_num < self.config.num_inputs:
                        if self.terminals[node_num].const:
                            node_map[node_num] = self.terminals[node_num].call([])
                        else:
                            node_map[node_num] = data_point[self.terminals[node_num].call([])]
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

        for position in range(self.config.num_genes - 1,
                              self.config.num_genesS - self.config.num_outputs - 1, -1):
            node_num = genome[position]
            if node_num >= self.config.num_inputs:
                nodes[node_num] = True

        for position in range(self.config.num_genes - self.config.num_outputs - 1, 0, -1):
            node_num = self.node_number(position)
            if (node_num in nodes.keys()
                    and self.phenotype(position) == self.GeneType.CONNECTION):
                gene = genome[position]
                if gene >= self.config.num_inputs:
                    nodes[gene] = True
        return sorted(nodes.keys())

    def decode(self, genome: list[int]):
        paths = []
        nodes = dict()
        for output_pos in range(self.config.num_genes - 1,
                                self.config.num_genes - self.config.num_outputs - 1, -1):
            nodes.clear()
            node_num = genome[output_pos]
            nodes[node_num] = True

            for gene_pos in range(self.config.num_genes - self.config.num_outputs - 1, 0, -1):
                if (self.node_number(gene_pos) in nodes.keys()
                        and self.phenotype(gene_pos) == self.GeneType.CONNECTION):
                    gene = genome[gene_pos]
                    nodes[gene] = True
            path = sorted(nodes.keys())
            paths.append(path)
        return paths

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
        if not self.hyperparameters.strict_selection:
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
            if random.random() < self.hyperparameters.mutation_rate:
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
            self.print_individual(individual)

    def print_individual(self, individual):
        print("Genome: " + str(individual[0]) + " : Fitness: " + str(individual[1]))

    def evolve(self):
        for job in range(self.config.num_jobs):
            best_fitness = None
            for generation in range(self.config.max_generations):
                self.breed()
                best_fitness = self.evaluate()
                print("Generation #" + str(generation) + " -> Best Fitness: " + str(best_fitness))
            print("Job #" + str(job) + " -> Best Fitness: " + str(best_fitness))


random.seed(SEED)
functions = [Add, Sub, Mul, Div]
terminals = [Var(0), Const(1)]
loss = euclidean_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate('KOZA1')

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
#problem  = BlackBoxProblem(data, actual, loss, IDEAL_FITNESS, MINIMIZING_FITNESS)
evaluator = GPAgent.evaluate_policy


problem = Problem(data, actual, loss, IDEAL_FITNESS, MINIMIZING_FITNESS)
config = CGPConfig(num_jobs=NUM_JOBS, max_generations=MAX_GENERATIONS, stopping_criteria=IDEAL_FITNESS,
                  minimizing_fitness=MINIMIZING_FITNESS, ideal_fitness = IDEAL_FITNESS,
                  silent_algorithm=SILENT_ALGORITHM, silent_evolver=SILENT_EVOLVER,
                  minimalistic_output=MINIMALISTIC_OUTPUT, num_functions = NUM_FUNCTIONS,
                  max_arity = MAX_ARITY, num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS,
                  num_function_nodes=NUM_FUNCTION_NODES, num_genes = NUM_GENES)
hyperparameters = CGPHyperparameters(mu=MU, lmbda=LAMBDA,
                                     population_size=POPULATION_SIZE,
                                     levels_back=LEVELS_BACK, mutation_rate=MUTATION_RATE,
                                     strict_selection=STRICT_SELECTION)
cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)

problem = PolicySearchProblem(env=env, model_=cgp, ideal_= IDEAL_FITNESS, minimizing_=False)

#cgp.evolve()
