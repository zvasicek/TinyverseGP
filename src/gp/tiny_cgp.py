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
    """
    Specialized hyperparameter space for CGP.
    """
    mu: int
    lmbda: int
    population_size: int
    levels_back: int
    strict_selection: bool
    mutation_rate: float = None
    mutation_rate_genes: int = None

@dataclass
class CGPConfig(GPConfig):
    """
    Specialized configuration to run CGP.
    """
    num_inputs: int
    num_outputs: int
    num_function_nodes: int
    num_functions: int
    max_arity: int
    max_time: int
    report_every_improvement: bool = False

    def init(self):
        self.genes_per_node = self.max_arity + 1
        self.num_genes = (self.genes_per_node * self.num_function_nodes)  + self.num_outputs

class Individual:
    """
    Class that is used to represent a CGP individual.
    A CGP individual is formally represented as a tuple consisting of
    the genome and the fitness value.
    """
    def __init__(self, genome: list[int], fitness: float = None):
        self.genome = genome
        self.fitness = fitness

class TinyCGP(GPModel):
    """
    Main class of the tiny CGP module that derives from GPModel and
    implements all related fundamental mechanisms tun run CGP.
    """
    class GeneType(Enum):
        """
        Enum for the gene type that are used for the CGP encoding
        """
        FUNCTION = 0
        CONNECTION = 1
        OUTPUT = 2

    class TerminalType(Enum):
        VARIABLE = 0
        CONSTANT = 1

    def __init__(self, problem_: Problem, functions_: list, terminals_: list,
                 config_: CGPConfig, hyperparameters_: Hyperparameters):
        self.num_evaluations = None
        self.population = []
        self.functions = functions_
        self.function2arity = [f.arity for f in functions_]
        self.terminals = terminals_
        self.problem = problem_
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.inputs = dict()
        self.init_inputs(terminals_)
        self.init_population()

    def init_population(self) -> list :
        """
        Initialization routine that creates and inits
        the individuals for the first generation.

        :return: list of individuals
        """
        self.population.clear()
        for _ in range(self.hyperparameters.population_size):
            individual = self.init_individual()
            self.population.append(individual)

    def init_individual(self) -> Individual:
        """
        Creates and initializes an individual.
        :return CGP individual
        """
        return Individual(self.init_genome())

    def init_inputs(self, terminals_: list):
        """
        Initializes the inputs by taking the passed terminals.
        """
        for index, terminal in enumerate(terminals_):
            if isinstance(terminal, Var):
                self.inputs[index] = (terminal, self.TerminalType.VARIABLE)
            else:
                self.inputs[index] = (terminal, self.TerminalType.CONSTANT)

    def init_genome(self) -> list[int]:
        """
        Initializes the genome by initializing each gene w.r.t. its type.
        :return: List of genes
        """
        return [self.init_gene(i) for i in range(self.config.num_genes)]

    def init_gene(self, position: int) -> int:
        """
        Initializes a gene at a specified position.

        :param position: position of the gene in the genotype.
        :return: gene
        """
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
        """
        Return the phenotype of a gene.
        :param position: gene position in the genome
        :return: gene type
        """
        if position >= self.config.num_function_nodes * (self.config.max_arity + 1):
            return self.GeneType.OUTPUT
        else:
            return self.GeneType.FUNCTION if position % (self.config.max_arity + 1) == 0 \
                else self.GeneType.CONNECTION

    def input_value(self, index: int) -> any:
        """
        Returns the input at a specified index.

        :param index: input index
        :return:input value
        """
        return self.inputs[index][0]

    def input_name(self, index: int) -> str:
        """
        Returns the name of an index at a specified index.

        :param index: input index
        :return:input name
        """
        idx = self.input_value(index)()
        return f'{self.input_value(index).name}({idx})' if idx is not None else self.input_value(index).name

    def input_type(self, index: int) -> TerminalType:
        """
        Returns the type of an index at a specified index.

        :param index: input index
        :return:input type
        """
        return self.inputs[index][1]

    def node_number(self, position: int) -> int:
        """
        Returns the node number at a specified position.

        :param position: position of the gene in the genome
        :return: node number where the gene belongs to
        """
        return math.floor(position / (self.config.max_arity + 1)) + self.config.num_inputs

    def node_position(self, node_num: int) -> int:
        """
        Returns the position of a node based on its number.

        :param node_num: number of the node in the genome
        :return: node position in the genome
        """
        return (node_num - self.config.num_inputs) * (self.config.max_arity + 1)

    def node_function(self, node_num: int, genome: list[int]) -> int:
        """
        Returns the function of the node at the given node number.

        :param node_num: Number of the node
        :param genome: Genome of the individual
        :return: function gene value
        """
        position = self.node_position(node_num)
        return genome[position]

    def node_connections(self, node_num: int, genome: list[int]) -> list:
        """
        Get the input connection genes of a node at the given node number.

        :param node_num: Number of the node
        :param genome: Genome of the individual
        :return: List connection gene values
        """
        position = self.node_position(node_num)
        inputs = []
        for count in range(self.config.max_arity):
            inputs.append(genome[position + count + 1])
        return inputs

    def outputs(self, genome: list[int]) -> list[int]:
        """
        Return the output genes.

        :param genome: Genome of an individual
        :return: List of output genes
        """
        outputs = []
        for output_pos in range(self.config.num_genes - 1,
                                self.config.num_genes - self.config.num_outputs - 1, -1):
            output = genome[output_pos]
            outputs.append(output)
        return outputs

    def max_gene(self, position: int):
        """
        Return the maximum gene value that is specified at a
        position in the genome.

        :param position: Gene position
        :return: Maximum gene value
        """
        if self.phenotype(position) == self.GeneType.OUTPUT:
            return self.config.num_inputs + self.config.num_functions - 1
        elif self.phenotype(position) == self.GeneType.CONNECTION:
            return self.config.num_functions - 1
        else:
            return self.node_number(position) - 1

    def fitness(self, individual: Individual) -> float:
        """
        Return the fitness value of an individual.
        """
        return individual.fitness

    def evaluate(self) -> (Individual, bool):
        """
        Evaluates the population.

        :returns: the best solution discovered in the population
        """
        best = None
        for individual in self.population:
            genome = individual.genome
            if (individual.fitness is None):
                individual.fitness = self.evaluate_individual(genome)
            fitness = individual.fitness

            if self.problem.is_ideal(fitness):
                return individual, True

            if best is None:
                best = individual
                best_fitness = fitness

            if self.problem.is_better(fitness, best_fitness):
                best = individual
                best_fitness = fitness

        return best, False

    def evaluate_individual(self, genome:list[int]) -> float:
        """
        Evaluate an individual against the problem.

        :param genome: the genome of an individual
        :return: fitness of the individual
        """
        self.num_evaluations += 1
        return self.problem.evaluate(genome, self)

    def evaluate_observation(self, genome: list[int], observation):
        """
        Evaluates an observation (of a dataset or environment)

        :param genome: Genome of an individual
        :param observation: Given observation
        :return: Prediction based on the genome and observation
        """
        paths = self.decode(genome)
        return self.predict(genome, observation, paths)

    def evaluate_node(self, node_num: int, genome: list[int], args: list) -> float:
        function = self.node_function(node_num, genome)
        arity = self.functions[function].arity
        if arity < len(args):
            args = args[:arity]
        return self.functions[function](*args)

    def predict_optimized_(self, genome: list[int], observation: list, paths=None) -> list:
        maxidx = self.config.genes_per_node * self.config.num_function_nodes
        step = self.config.genes_per_node
        
        node_values = [None for i in range(self.config.num_function_nodes + self.config.num_inputs)]
        for i in range(self.config.num_inputs):
            if not self.terminals[i].const:
                node_values[i] = observation[i]
            else:
                node_values[i] = self.terminals[i]()

        nidx = self.config.num_inputs
        for node_num in range(0, maxidx, step):
            fun = genome[node_num]
            args = [node_values[i] for i in genome[node_num + 1:node_num + self.function2arity[fun] + 1]]
            node_values[nidx] = self.functions[fun](*args)
            nidx += 1
        
        return [node_values[genome[maxidx+i]] for i in range(self.config.num_outputs)]

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
                            node_map[node_num] = self.terminals[node_num]()
                        else:
                            node_map[node_num] = observation[self.terminals[node_num]()]
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

    def breed(self, parent : Individual = None):
        self.population.clear()
        self.population.append(parent)
        for _ in range(self.hyperparameters.lmbda):
            # mutation requires fitness calculation, so we need to genome only
            offspring = Individual(parent.genome.copy()) 
            self.mutation(offspring.genome)
            self.population.append(offspring)

    def selection(self) -> list:
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=not self.config.minimizing_fitness)
        count = 0
        if not self.hyperparameters.strict_selection:
            best_fitness = sorted_pop[0].fitness
            for individual in sorted_pop:
                if individual.fitness != best_fitness:
                    break
                else:
                    count += 1
            parent = random.randint(0, count - 1)
        else:
            parent = 0
        return sorted_pop[parent]

    def mutation(self, genome: list[int]):
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

        def generate_expr_map(genome: list[int], active_nodes=None):
            if active_nodes is None:
                active_nodes = self.active_nodes(genome)
            expr_map = dict()
            for node_num in active_nodes:
                function = self.node_function(node_num, genome)
                node_arity = self.function2arity[function]
                args = self.node_connections(node_num, genome)[0:node_arity]
                func_name = self.functions[function].name
                node_expr = func_name + "("
                for index, argument in enumerate(args):
                    if argument in expr_map:
                        arg_expr = expr_map[argument]
                    else:
                        arg_expr = self.input_name(argument)
                    node_expr += arg_expr
                    if index < node_arity - 1:
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
        print(f'Genome: {individual.genome} Fitness: {individual.fitness}')

    def evolve(self):
        last_fitness = None
        t0 = time.time()
        elapsed = 0
        terminate = False
        silent = self.config.silent_algorithm
        rc = self.config.report_every_improvement
        for job in range(self.config.num_jobs):
            self.num_evaluations = 0
            # evaluate the initial population
            best_individual,_ = self.evaluate()
            best_fitness = best_fitness_job = best_individual.fitness
            for generation in range(self.config.max_generations):
                # Selection of a parent if necessary
                parent = best_individual
                if not self.hyperparameters.strict_selection:
                    parent = self.selection()
                
                # Population breeding
                self.breed(parent)
                
                # Evaluation of the offspring
                best_gen, is_ideal = self.evaluate()
                best_gen_fitness = best_gen.fitness

                if self.problem.is_better(best_gen_fitness, best_fitness):
                    best_individual = best_gen
                    best_fitness = best_gen_fitness

                if self.problem.is_better(best_gen_fitness, best_fitness_job):
                    best_fitness_job = best_gen_fitness

                if (not silent) and ((rc and (best_fitness != last_fitness)) or (self.config.report_interval and (generation % self.config.report_interval == 0))):
                    last_fitness = best_fitness
                    self.report_generation(silent=False,
                                        generation=generation,
                                        best_fitness=best_fitness,
                                        report_interval=1)
                
                if is_ideal: # if the ideal solution is found, terminate    
                    break

                if (generation & 15) == 0: # check periodically if the time limit is reached
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
        return best_individual
