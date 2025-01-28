# GPBench TinyCGP 0.1
# Roman Kalkreuth (roman.kalkreuth@rwth-aachen.de)
# RWTH Aachen University (Germany)

__author__ = 'Roman Kalkreuth'
__version__ = '0.1'
__email__ = 'roman.kalkreuth@rwth-aachen.de'

import math
import random
import copy
#import numpy as np
import time

from dataclasses import dataclass
from enum import Enum

from src.gp.tinyverse import GPModel, Hyperparameters, GPConfig, Var
from src.gp.problem import Problem

@dataclass
class CGPHyperparameters(Hyperparameters):
    '''
    Hyperparameters for CGP 
    '''
    mu: int
    lmbda: int
    population_size: int
    levels_back: int
    strict_selection: bool
    mutation_rate: float = None
    mutation_rate_genes: int = None

@dataclass
class CGPConfig(GPConfig):
    '''
    Configuration for CGP 
    '''
    num_inputs: int
    num_outputs: int
    num_function_nodes: int
    num_functions: int
    max_arity: int
    max_time: int

    def init(self):
        self.num_genes = ((self.max_arity + 1) * self.num_function_nodes)  + self.num_outputs

class TinyCGP(GPModel):
    '''
    TinyCGP class
    '''

    class GeneType(Enum):
        '''
        Enum for gene types
        '''
        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        '''
        Enum for terminal types
        '''
        VARIABLE = 0
        CONSTANT = 1

    def __init__(self, problem_: Problem, functions_: list, terminals_: list,
                 config_: CGPConfig, hyperparameters_: Hyperparameters):
        self.num_evaluations = None
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
        '''
        Initialize the population. 
        '''
        for _ in range(self.hyperparameters.population_size):
            individual = self.init_individual()
            self.population.append(individual)

    def init_individual(self):
        '''
        Initialize an individual.
        '''
        genome = self.init_genome()
        fitness = 0.0
        return [genome, fitness]

    def init_inputs(self, terminals_: list):
        '''
        Initialize the inputs.
        '''
        for index, terminal in enumerate(terminals_):
            if isinstance(terminal, Var):
                self.inputs[index] = (terminal, self.TerminalType.VARIABLE)
            else:
                self.inputs[index] = (terminal, self.TerminalType.CONSTANT)

    def init_genome(self) -> list[int]:
        '''
        Initialize a genome.
        '''
        genome = [0 for i in range(self.config.num_genes)]
        for count in range(self.config.num_genes):
            gene = self.init_gene(count)
            genome[count] = gene
        return genome

    def init_gene(self, position: int) -> int:
        '''
        Initialize a gene.
        '''
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
        '''
        Determine the phenotype of a gene.
        '''
        if position >= self.config.num_function_nodes * (self.config.max_arity + 1):
            return self.GeneType.OUTPUT
        else:
            return self.GeneType.FUNCTION if position % (self.config.max_arity + 1) == 0 \
                else self.GeneType.CONNECTION

    def input_value(self, index: int) -> any:
        '''
        Get the input value.
        '''
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        '''
        Get the input name.
        '''

        return str(self.input_value(index).name) + "(" + str(self.input_value(index).call([]))+ ")"

    def input_type(self, index: int) -> TerminalType:
        '''
        Get the input type.
        '''
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        '''
        Get the node number.
        '''
        return math.floor(position / (self.config.max_arity + 1)) + self.config.num_inputs

    def node_position(self, node_num: int) -> int:
        '''
        Get the node position.
        '''
        return (node_num - self.config.num_inputs) * (self.config.max_arity + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        '''
        Get the node function.
        '''
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> []:
        '''
        Get the node connections.
        '''
        position = self.node_position(node_num)
        inputs = []
        for count in range(self.config.max_arity):
            inputs.append(genome[position + count + 1])
        return inputs

    def outputs(self, genome: list[int]) -> list[int]:
        '''
        Get the outputs.
        '''
        outputs = []
        for output_pos in range(self.config.num_genes - 1,
                                self.config.num_genes - self.config.num_outputs - 1, -1):
            output = genome[output_pos]
            outputs.append(output)
        return outputs

    def max_gene(self, position: int):
        '''
        Get the maximum gene.
        '''
        if self.phenotype(position) == self.GeneType.OUTPUT:
            return self.config.num_inputs + self.config.num_functions - 1
        elif self.phenotype(position) == self.GeneType.CONNECTION:
            return self.config.num_functions - 1
        else:
            return self.node_number(position) - 1

    def fitness(self, individual: list[list[int], float]):
        '''
        Get the fitness of an individual.
        '''
        return individual[1]

    def evaluate(self) -> tuple:
        '''
        Evaluate the population.
        '''
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
                best_genome = copy.deepcopy(individual[0])
                best_fitness = individual[1]
        return best_genome, best_fitness, False

    def evaluate_individual(self, genome:list[int]) -> float:
        '''
        Evaluate an individual.
        '''
        self.num_evaluations += 1
        return self.problem.evaluate(genome, self)

    def evaluate_observation(self, genome: list[int], observation):
        '''
        Evaluate an observation.
        '''
        paths = self.decode(genome)
        return self.predict(genome, observation, paths)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        '''
        Evaluate a node.
        '''
        function = self.node_function(node_num, genome)
        arity = self.functions[function].arity
        if arity < len(args):
            args = args[:arity]
        return self.functions[function].call(args)

    def predict(self, genome: list[int], observation: list, paths=None) -> list:
        '''
        Predict the output for a single observation.
        '''
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
        '''
        Get the current active nodes of the genome.
        '''
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
        '''
        Decode the genome.
        '''
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
        '''
        Breed the population.
        '''
        parent = self.selection()
        self.population.clear()
        self.population.append(parent)
        for _ in range(self.hyperparameters.lmbda):
            offspring = copy.deepcopy(parent)
            self.mutation(offspring[0])
            self.population.append(offspring)

    def selection(self) -> list:
        '''
        Select a random individual among the best solutions.
        '''
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
        '''
        Mutates the genome 
        '''
        if self.hyperparameters.mutation_rate is not None:
            for index, gene in enumerate(genome):
                if random.random() < self.hyperparameters.mutation_rate:
                    genome[index] = self.init_gene(index)
        elif self.hyperparameters.mutation_rate_genes is not None:
            mg = random.randint(1, self.hyperparameters.mutation_rate_genes)
            for _ in range(mg):
                index = random.randint(0, self.config.num_genes - 1)
                genome[index] = self.init_gene(index)
        else:
            raise ValueError("Either mutation_rate or mutation_rate_genes must be set")

    def expression(self, genome: list[int]) -> list[str]:
        '''
        Returns the expression of the genome as a list of strings, one for each output.
        '''

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
                    if index < self.config.max_arity - 1:
                        node_expr += ", "
                node_expr += ")"
                expr_map[node_num] = node_expr
            return expr_map

        expr_map = generate_expr_map(genome)
        expressions = []

        outputs = self.outputs(genome)

        for output in outputs:
            if output < self.config.num_inputs:
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
        '''
        Perform the evolutionary procedure.
        '''
        best_solution = None
        best_fitness = None
        t0 = time.time()
        elapsed = 0
        terminate = False
        for job in range(self.config.num_jobs):
            best_fitness_job = None
            self.num_evaluations = 0
            for generation in range(self.config.max_generations):
                self.breed()
                best_solution_gen, best_fitness_gen, is_ideal = self.evaluate()

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
                t1 = time.time()
                delta = t1 - t0
                t0 = t1
                elapsed += delta
                if elapsed + delta >= self.config.max_time:
                    terminate = True
                    break


            self.report_job(job = job,
                            num_evaluations=self.num_evaluations,
                            best_fitness=best_fitness_job,
                            silent_evolver=self.config.silent_evolver,
                            minimalistic_output=self.config.minimalistic_output)
            if terminate:
                break
        return best_solution
