# GPBench TinyCGP 0.1
# Roman Kalkreuth (roman.kalkreuth@rwth-aachen.de)
# RWTH Aachen University (Germany)

__author__ = 'Roman Kalkreuth'
__version__ = '0.1'
__email__ = 'roman.kalkreuth@rwth-aachen.de'

import math
import random
import copy
import numpy as np

from dataclasses import dataclass
from enum import Enum
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import ale_py

from src.gp.tinyverse import Var, Const, GPModel, Hyperparameters, GPConfig
from functions import ADD, SUB, MUL, DIV, AND, OR, NOT, NAND, NOR, LT, LTE, GT, GTE, EQ, MIN, MAX, NEG, IF, IFGTZ, IFLEZ
from loss import euclidean_distance
from src.gp.problem import Problem, BlackBox, PolicySearch

MU = 1
LAMBDA = 4
POPULATION_SIZE = MU + LAMBDA
NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_FUNCTION_NODES = 20
MAX_ARITY = 3
NUM_NODES = NUM_INPUTS + NUM_FUNCTION_NODES + NUM_OUTPUTS
LEVELS_BACK = NUM_FUNCTION_NODES
NUM_FUNCTIONS = 4
MUTATION_RATE = 0.1
MAX_GENERATIONS = 500
NUM_JOBS = 1
IDEAL_FITNESS = 500
STRICT_SELECTION = False
MINIMIZING_FITNESS = False
SEED = 42

SILENT_ALGORITHM = False
SILENT_EVOLVER = False
MINIMALISTIC_OUTPUT = False
REPORT_INTERVAL = 50

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
    num_functions: int
    max_arity: int

    def init(self):
        self.num_genes = ((self.max_arity + 1) * self.num_function_nodes)  + self.num_outputs

def check_config():
    if LEVELS_BACK > NUM_FUNCTION_NODES:
        raise ValueError('LEVELS_BACK > NUM_FUNCTION_NODES')

class TinyCGP(GPModel):

    class GeneType(Enum):
        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        VARIABLE = 0
        CONSTANT = 1

    def __init__(self, problem_: Problem, functions_: list, terminals_: list,
                 config_: GPConfig, hyperparameters_: Hyperparameters):
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
        genome = np.zeros(self.config.num_genes, dtype=np.uint32)
        for count in range(self.config.num_genes):
            gene = self.init_gene(count)
            genome[count] = gene
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
            rand = random.randint(0, self.config.num_inputs + self.config.num_function_nodes - 1)
            return rand

    def phenotype(self, position: int) -> GeneType:
        if position >= self.config.num_function_nodes * (self.config.max_arity + 1):
            return self.GeneType.OUTPUT
        else:
            return self.GeneType.FUNCTION if position % (self.config.max_arity + 1) == 0 \
                else self.GeneType.CONNECTION

    def input_value(self, index: int) -> any:
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        return str(self.input_value(index).name)

    def input_type(self, index: int) -> TerminalType:
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        return math.floor(position / (self.config.max_arity + 1)) + self.config.num_inputs

    def node_position(self, node_num: int) -> int:
        return (node_num - self.config.num_inputs) * (self.config.max_arity + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> []:
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

    def evaluate(self) -> tuple:
        best_genome = None
        best_fitness = None
        for individual in self.population:
            genome = individual[0]
            fitness = individual[1] = self.evaluate_individual(genome)

            if best_genome is None:
                best_genome = genome
                best_fitness = fitness

            if self.problem.is_ideal(fitness):
                return genome, fitness, True

            if self.problem.is_better(fitness, best_fitness):
                best_genome = individual
                best_fitness = individual[1]
        return best_genome, best_fitness, False

    def evaluate_individual(self, genome:list[int]) -> float:
        self.num_evaluations += 1
        return self.problem.evaluate(genome, self)

    def evaluate_observation(self, genome: list[int], observation):
        paths = self.decode(genome)
        return self.predict(genome, observation, paths)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        function = self.node_function(node_num, genome)
        arity = self.functions[function].arity
        if arity < len(args):
            args = args[:arity]
        return self.functions[function].call(args)

    def predict(self, genome: list[int], observation: list, paths=None) -> list:
        node_map = dict()
        prediction = []

        if paths is None:
            paths = self.decode(genome)

        for count, path in enumerate(paths):
            cost = 0.0
            for node_num in path:
                if node_num not in node_map.keys():
                    if node_num < self.config.num_inputs:
                        if self.terminals[node_num].const:
                            node_map[node_num] = self.terminals[node_num].call([])
                        else:
                            node_map[node_num] = observation[self.terminals[node_num].call([])]
                    else:
                        arguments = []
                        connections = self.node_connections(node_num, genome)
                        for connection in connections:
                            argument = node_map[connection]
                            arguments.append(argument)
                        node_map[node_num] = self.evaluate_node(node_num, genome, arguments)
                cost = node_map[node_num]
            prediction.append(cost)
        return prediction

    def active_nodes(self, genome: list[int]):
        nodes = dict()

        for position in range(self.config.num_genes - 1,
                              self.config.num_genes - self.config.num_outputs - 1, -1):
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
        sorted_pop = sorted(self.population, key=lambda ind: ind[1], reverse=not self.config.minimizing_fitness)
        count = 0
        if not self.hyperparameters.strict_selection:
            best_fitness = sorted_pop[0][1]
            for individual in sorted_pop:
                if individual[1] != best_fitness:
                    break
                else:
                    count += 1
            parent = random.randint(0, count - 1)
        else:
            parent = 0
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
                func_name = self.functions[function].name
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
        best_solution = None
        best_fitness = None
        for job in range(self.config.num_jobs):
            best_fitness_job = None
            self.num_evaluations = 0
            for generation in range(self.config.max_generations):
                self.breed()
                best_solution_gen, best_fitness_gen, is_ideal = self.evaluate()
                #best_solution_gen, best_fitness_gen = self.population[0][0], self.population[0][1]

                if best_solution is None or self.problem.is_better(best_fitness_gen, best_fitness):
                    best_solution = best_solution_gen
                    best_fitness = best_fitness_gen

                if best_fitness_job is None or self.problem.is_better(best_fitness_gen, best_fitness_job):
                    best_fitness_job = best_fitness_gen

                self.report_generation(silent = self.config.silent_algorithm,
                                       generation=generation,
                                       best_fitness=best_fitness,
                                       report_interval=self.config.report_interval)
                if is_ideal:
                    break

            self.report_job(job = job,
                            num_evaluations=self.num_evaluations,
                            best_fitness=best_fitness_job,
                            silent_evolver=self.config.silent_evolver,
                            minimalistic_output=self.config.minimalistic_output)
        return best_solution


#random.seed(SEED)

env = gym.make("ALE/Breakout-v5",)
wrapped_env = FlattenObservation(env)

NUM_INPUTS = wrapped_env.observation_space.shape[0]
NUM_OUTPUTS = 5

functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, LTE, GT, GTE, EQ, NEG, MIN, MAX, IF, IFGTZ, IFLEZ]
terminals = [Var(index=count) for count in range(NUM_INPUTS)]

config = CGPConfig(num_jobs=NUM_JOBS, max_generations=MAX_GENERATIONS, stopping_criteria=IDEAL_FITNESS,
                  minimizing_fitness=MINIMIZING_FITNESS, ideal_fitness = IDEAL_FITNESS,
                  silent_algorithm=SILENT_ALGORITHM, silent_evolver=SILENT_EVOLVER, report_interval = REPORT_INTERVAL,
                  minimalistic_output=MINIMALISTIC_OUTPUT, num_functions = len(functions),
                  max_arity = MAX_ARITY, num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS,
                  num_function_nodes=NUM_FUNCTION_NODES)

config.init()

hyperparameters = CGPHyperparameters(mu=MU, lmbda=LAMBDA,
                                     population_size=POPULATION_SIZE,
                                     levels_back=LEVELS_BACK, mutation_rate=MUTATION_RATE,
                                     strict_selection=STRICT_SELECTION)

problem = PolicySearch(env=env, ideal_= IDEAL_FITNESS, minimizing_=MINIMIZING_FITNESS, num_episodes_= 10)
#problem  = BlackBoxProblem(data, actual, loss, IDEAL_FITNESS, MINIMIZING_FITNESS)
cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)
best_solution = cgp.evolve()
#print(best_solution)
env.close()


#expression = cgp.expression(best_solution)
#print(expression)

#render_mode="human"
#render_mode = "rgb_array"
# LunarLander-v3
gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")
#env = gym.wrappers.RecordVideo(env, 'video')
#env = RecordVideo(env, video_folder="gym-recordings", name_prefix="lunar-lander")
problem = PolicySearch(env=env, ideal_= 100, minimizing_=MINIMIZING_FITNESS)
problem.evaluate(best_solution, cgp, num_episodes = 5)
env.close()


#loss = euclidean_distance
#benchmark = SRBenchmark()
#data, actual = benchmark.generate('KOZA1')