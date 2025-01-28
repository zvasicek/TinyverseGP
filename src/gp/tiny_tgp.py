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
import time

from abc import ABC
from abc import abstractmethod
from typing import List, Any
from dataclasses import dataclass

from src.gp.tinyverse import GPModel, GPConfig, GPHyperparameters
from src.gp.functions import *
from src.gp.problem import *
from src.gp.tinyverse import *
from src.benchmark.benchmark import *

class Node:
    '''
    Node class for the tree representation of the genome.
    '''
    def __init__(self, function: Any, children: List[Any]):
        self.function = function
        self.children = children

def node_size(node: Node) -> int:
    '''
    Return the number of nodes in a tree.
    '''
    if len(node.children) == 0:
        return 1
    return 1 + sum([node_size(child) for child in node.children])


class TinyTGP(GPModel):
    '''
    Implementation of a Tiny Tree-based Genetic Programming model.
    '''
    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: list[Function]

    def __init__(self, problem_: object, functions_: list[Function], terminals_: list[Function], config: Config, hyperparameters: Hyperparameters):
        self.functions = functions_ # [f for f in functions_ if f.arity > 0]
        self.terminals = terminals_ # [f for f in functions_ if f.arity == 0]
        self.problem = problem_
        self.hyperparameters = hyperparameters
        self.config = config
        self.best = None
        self.num_evaluations = 0
        self.population = [[genome, 0.0] 
                            for genome in self.init_ramped_half_half(self.hyperparameters.pop_size, 1, self.hyperparameters.max_depth, self.hyperparameters.max_size)]
        self.evaluate()


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
        '''
        Initialize the population with the ramped half-and-half method.
        '''
        pop = []
        for md in range(min_depth, max_depth + 1):
            grow = True
            for _ in range(int(num_pop / (max_depth - 3 + 1))):
                trees = []
                for _ in range(self.config.num_outputs):
                    if grow:
                        tree = self.tree_random_grow(min_depth, max_depth, max_size)
                    else:
                        tree = self.tree_random_full(max_depth, max_size)
                    trees.append(tree)
                pop.append(trees)
                grow = not grow
        return pop

    def evaluate(self) -> float:
        '''
        Evaluate the population.
        '''
        best = None
        for ix, individual in enumerate(self.population):
            genome = individual[0]
            fitness = self.evaluate_individual(genome)
            self.population[ix][1] = fitness

            if best is None or self.problem.is_better(fitness, best):
                best = fitness
            if self.best is None or self.problem.is_better(fitness, self.best[1]):
                self.best = [copy.copy(genome), fitness]
        return best

    def evaluate_individual(self, genome:list[int]) -> float:
        '''
        Evaluate a single individual.
        '''
        self.num_evaluations += 1
        f = self.problem.evaluate(genome, self)
        if self.best is None or self.problem.is_better(f, self.best[1]):
            self.best = [copy.copy(genome), f]
        return f

    def predict(self, genome: Node, observation: list) -> list:
        '''
        Predict the output of the genome given the observation.
        '''
        def eval_node(node: Node):
            if node.function.arity == 0:
                if node.function.const:
                    return node.function()
                else:
                    return observation[node.function()]
            else:
                args = [eval_node(child) for child in node.children]
                return node.function(*args)

        return [eval_node(g) for g in genome]

    def breed(self):
        '''
        Breed the population by first selecting a set of pair of parents and then
        applying crossover and mutation operators.
        '''
        parents = [[self.selection(), self.selection()] for _ in range(self.hyperparameters.pop_size-1)]
        self.population = [self.perturb(*parent) for parent in parents]
        self.population.append([copy.copy(self.best[0]), self.best[1]])

    def perturb(self, parent1: Node, parent2: Node) -> list:
        '''
        Applies the crossover and mutation operators to the parents.
        '''
        genome = self.crossover(parent1, parent2) if random.random() <= self.hyperparameters.cx_rate else parent1
        genome = self.mutation(genome, self.hyperparameters.max_depth,
                               self.hyperparameters.max_size) if random.random() <= self.hyperparameters.mutation_rate else genome
        return [genome, None]

    def selection(self) -> Node:
        '''
        Select a parent from the population using the tournament selection method.
        '''
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        if self.problem.minimizing:
            return min(parents, key=lambda ind: ind[1])[0]
        else:
            return max(parents, key=lambda ind: ind[1])[0]

    def crossover(self, p1: list, p2: list) -> list:
        '''
        Apply the crossover operator to the parents.
        '''
        ix = random.choice(range(self.config.num_outputs))
        n = [copy.copy(p) for p in p1]
        n[ix] = self.subtree_crossover(p1[ix], p2[ix])
        return n

    def subtree_crossover(self, p1: Node, p2: Node) -> Node:
        '''
        Apply the subtree crossover operator to the parents.
        '''
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

    def mutation(self, n: list, max_depth: int, size: int):
        '''
        Apply the mutation operator to the parent.
        '''
        ix = random.choice(range(self.config.num_outputs))
        new_n = copy.deepcopy(n)
        n[ix] = self.subtree_mutation(n[ix], max_depth, size)
        return n

    def subtree_mutation(self, n: Node, max_depth: int, size: int):
        '''
        Apply the subtree mutation operator to the parent.
        '''
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

    def expression(self, genome: list) -> list[str]:
        '''
        Return the expression of the genome as a string.
        '''

        def print_node(node: Node):
            if len(node.children) == 0:
                return node.function.name + "(" + str(node.function()) + ")"
            else:
                args = [print_node(child) for child in node.children]
                return node.function.name + "(" + ", ".join(args) + ")"

        return [print_node(g) for g in genome]

    def print_population(self):
        for individual in self.population:
            self.print_individual(individual)

    def print_individual(self, individual):
        print("Genome: " + ";".join(self.expression(individual[0])) + " : Fitness: " + str(individual[1]))

    def evolve(self):
        # NOTE: is this supposed as a preparation for multithreading?
        t0 = time.time()
        elapsed = 0
        terminate = False
        best_fitness_job = None
        for job in range(self.config.num_jobs):
            best_fitness = None
            for generation in range(self.config.max_generations):
                self.breed()
                best_fitness = self.evaluate()
                #print("Generation #" + str(generation) + " -> Best Fitness: " + str(best_fitness))
                self.report_generation(silent = self.config.silent_algorithm,
                                       generation=generation,
                                       best_fitness=best_fitness,
                                       report_interval=self.config.report_interval)
                t1 = time.time()
                delta = t1-t0
                t0 = t1
                elapsed += delta
                if elapsed + delta >= self.config.max_time:
                    terminate = True
                    break
            #self.print_individual(self.best)
            if best_fitness_job is None or self.problem.is_better(best_fitness, best_fitness_job):
                    best_fitness_job = best_fitness
            #print("Job #" + str(job) + " -> Best Fitness: " + str(best_fitness))
            self.report_job(job = job,
                            num_evaluations=self.num_evaluations,
                            best_fitness=best_fitness_job,
                            silent_evolver=self.config.silent_evolver,
                            minimalistic_output=self.config.minimalistic_output)
            if terminate:
                break
        return self.best[0]
