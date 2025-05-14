"""
TinyGP: A minimalistic implementation of tree-based Genetic Programming for
         TinyverseGP.
"""

import random
import copy
import time
from src.gp.problem import *
from src.gp.tinyverse import *

class Node:
    '''
    Node class for the tree representation of the genome.

    Each node of a tree is represented as a function and a list of Nodes, that
    will serve as the arguments to that function.
    NOTE: the list of children must be of the same arity of the function.
    '''
    def __init__(self, function: Any, children: List[Any]):
        self.function = function
        self.children = children

@dataclass
class TGPHyperparameters(GPHyperparameters):
    """
    Specialized hyperparameter configuration space for TGP.
    """
    max_size: int
    max_depth: int

    def __post_init__(self):
        GPHyperparameters.__post_init__(self)

class TGPIndividual(GPIndividual):
    genome: list[Node]
    fitness: any

    def __init__(self, genome_: list[int], fitness_: any = None):
        GPIndividual.__init__(self,genome_, fitness_)

def node_size(node: Node) -> int:
    '''
    Return the number of nodes in a tree.
    '''
    if len(node.children) == 0:
        return 1
    return 1 + sum([node_size(child) for child in node.children])


class TinyTGP(GPModel):
    '''
    Main class of the tiny TGP module that derives from GPModel and
    implements all related fundamental mechanisms to run TGP.
    '''
    config: GPConfig
    hyperparameters: TGPHyperparameters
    problem: Problem
    functions: list[Function]

    def __init__(self, problem_: Problem, functions_: list[Function], terminals_: list[Function],
                 config: GPConfig, hyperparameters: TGPHyperparameters):
        self.functions = functions_ # the list of available functions
        self.terminals = terminals_ # the list of terminal nodes
        self.problem = problem_ # an instance to a problem. This allows us to handle different problems transparently
        self.hyperparameters = hyperparameters # hyperparameters
        self.config = config # overall configuration
        self.best_individual = None  # to keep the best program found so far
        self.num_evaluations = 0 # conter of number of evaluations
        # initial population using ramped half-and-half
        self.population = [TGPIndividual(genome, 0.0)
                            for genome in self.init_ramped_half_half(self.hyperparameters.pop_size, 1,
                                                                     self.hyperparameters.max_depth,
                                                                     self.hyperparameters.max_size)]
        self.evaluate() # evaluates the initial population


    def tree_random_full(self, max_depth: int, size: int) -> Node:
        """
        Generates a random tree using the full method limited by a `max_depth` and `size`.

        :returns: `Node`
        """
        # if we reached the maximum depth or there are only two or less nodes available
        # according to size, we sample a terminal node.
        if max_depth == 0 or size < 2:
            return Node(random.choice(self.terminals), [])
        # otherwise we sample a non-terminal node and generate the children recursively,
        # reducing the depth by one and splitting the maximum size available by all children
        n = random.choice(self.functions)
        children = [self.tree_random_full(max_depth - 1, size // n.arity - 1) for _ in range(n.arity)]
        return Node(n, children)

    def tree_random_grow(self, min_depth: int, max_depth: int, size: int) -> Node:
        """
        Generates a random tree using the grow method limited by a `min_depth`, `max_depth` and `size`.

        :returns: `Node`
        """
        # if we cannot add more non-terminals, sample a terminal
        if max_depth <= 1 or size < 2:
            return Node(random.choice(self.terminals), [])
        # if we are already at the minimum depth, sample a terminal with 50% chance                 
        if min_depth <= 0 and random.random() < 0.5:
            return Node(random.choice(self.terminals), [])
        else:
            # Let's sample a non-terminal and generate
            # n.arity children calling `tree_random_grow` recursively and adjusting the maximum depth and size.
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
        It will create one tree per output.

        Creates an almost equal number of trees from min_depth to max_depth
        while taking turns between grow and full method.
        This is supposed to ensure a variability of tree sizes and balance.

        :return: a list of lists of `Node`
        '''
        pop = []
        for md in range(min_depth, max_depth + 1):
            grow = True
            for _ in range(int(num_pop / (max_depth - 3 + 1))):
                # the individual may be represented by multiple trees
                # if the problem requires multiple outputs
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
        Triggers the evaluation of the whole population.

        :return: a `float` value of the best fitness
        '''
        best = None
        # For each individual in the population 
        for ix, individual in enumerate(self.population):
            genome = individual.genome  # extract the genome
            fitness = self.evaluate_individual(genome) # evaluate it
            self.population[ix] = TGPIndividual(genome, fitness) # assign the fitness
            # update the population best solution
            if best is None or self.problem.is_better(fitness, best):
                best = fitness
            # update the best solution of all time
            if self.best_individual is None or self.problem.is_better(fitness, self.best_individual.fitness):
                self.best_individual = TGPIndividual(genome, fitness)
        return best

    def evaluate_individual(self, genome:list[int]) -> float:
        '''
        Evaluate a single individual `genome`.

        :return: a `float` representing the fitness of that individual.
        '''
        self.num_evaluations += 1  # update the evaluation counter
        f = self.problem.evaluate(genome, self) # evaluate the solution using the problem instance
        if self.best_individual is None or self.problem.is_better(f, self.best_individual.fitness):
            self.best_individual = TGPIndividual(genome, f)
        return f

    def predict(self, genome: Node, observation: list) -> list:
        '''
        Predict the output of the `genome` given a single `observation`.

        :return: a list of the outputs for that observation
        '''
        def eval_node(node: Node):
            if node.function.arity == 0:
                # if the node is a constant, return its value
                if node.function.const:
                    return node.function()
                else:
                    # if it is a variable, returns the corresponding value from the observation
                    return observation[node.function()]
            else:
                # if it is a function, firt evaluate the children and then
                # call the function passing the evaluated children as arguments.
                args = [eval_node(child) for child in node.children]
                return node.function(*args)

        # if the genome is a multi-tree (to induce multiple outputs), apply eval_node to each tree
        return [eval_node(g) for g in genome]

    def breed(self):
        '''
        Breed the population by first selecting a set of pair of parents and then
        applying crossover and mutation operators.
        '''
        # Select n pairs of parents using tournament selection, n is the population size minus 1 (so we have space for the best individual)
        parents = [[self.selection(), self.selection()] for _ in range(self.hyperparameters.pop_size-1)]
        # replace the current population by perturbing the sampled parents     
        self.population = [self.perturb(*parent) for parent in parents]
        # keep the best solution in the population 
        self.population.append(TGPIndividual(self.best_individual.genome, self.best_individual.fitness))

    def perturb(self, parent1: Node, parent2: Node) -> list:
        '''
        Applies the crossover and mutation operators to the parents.

        :return: a list of the `genome` and `None` representing the unevaluated fitness.
        '''
        # applies the crossover with `self.hyperparameters.cx_rate` probability, otherwise return the first parent
        genome = self.crossover(parent1, parent2) if random.random() <= self.hyperparameters.cx_rate else parent1
        # applies mutation with `self.hyperparameters.mutation_rate` probability to the current offspring
        genome = self.mutation(genome, self.hyperparameters.max_depth,
                               self.hyperparameters.max_size) if random.random() <= self.hyperparameters.mutation_rate else genome
        return TGPIndividual(genome, None) # returns the unevaluated offspring

    def selection(self) -> Node:
        '''
        Select a parent from the population using the tournament selection method.

        :return: a selected `Node` from the population.
        '''
        # samples `self.hyperparameters.tournament_size` solutions completely at random
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        # return the best of this sample whether it is a minimization or maximization problem     
        if self.problem.minimizing:
            return min(parents, key=lambda ind: ind.fitness).genome
        else:
            return max(parents, key=lambda ind: ind.fitness).genome

    def crossover(self, p1: list, p2: list) -> list:
        '''
        Apply the crossover operator to the parents. For multiple trees (i.e., multiple outputs)
        it will choose one tree at random.

        :return: the recombined trees
        '''
        # chose the tree to apply crossover if we have multiple trees     
        ix = random.choice(range(self.config.num_outputs))
        n = [copy.copy(p) for p in p1]
        n[ix] = self.subtree_crossover(p1[ix], p2[ix])
        return n

    def subtree_crossover(self, p1: Node, p2: Node) -> Node:
        '''
        Apply the subtree crossover operator to the parents.

        :return: the recombined `Node`
        '''
        def pick_from(n: Node, ix: int) -> Node:
            # if we reached the desired node, return it     
            if ix == 0:
                return n
            tryout = None
            ix = ix - 1
            # for each children     
            for iy in range(n.function.arity):
                # try to pick the specified node
                tryout = pick_from(n.children[iy], ix)
                ix = ix - node_size(n.children[iy])
                # if we found it, break the loop and return it, otherwise, keep searching     
                if tryout is not None:
                    break
            return tryout

        def assemble(n1: Node, n2: Node, ix: int) -> Node:
            # if we found the node we want to replace, return the replacement piece     
            if ix == 0:
                return n2
            # otherwise, copy the node and call assemble to copy the children of n1         
            new_node = copy.deepcopy(n1)
            children = []
            ix = ix - 1
            for child in new_node.children:
                children.append(assemble(child, n2, ix) if ix > 0 else child)
                ix = ix - node_size(child)
            return Node(n1.function, children)
        # extract the subtree from the second parent at a random node
        piece2 = pick_from(p2, random.choice(range(node_size(p2))))
        # replace the subtree of p1 at a random point with `piece2`
        return assemble(p1, piece2, random.choice(range(node_size(p1))))

    def mutation(self, n: list, max_depth: int, size: int):
        '''
        Apply the mutation operator to the parent. For multiple trees (i.e., multiple outputs)
        it will choose one tree at random.

        :return: the mutated tree
        '''
        ix = random.choice(range(self.config.num_outputs))
        new_n = copy.deepcopy(n)
        n[ix] = self.subtree_mutation(n[ix], max_depth, size)
        return n

    def subtree_mutation(self, n: Node, max_depth: int, size: int):
        '''
        Apply the subtree mutation operator to the parent.

        :return: the mutate `Node`
        '''
        # pick a random node     
        n_nodes = node_size(n)
        ix = random.choice(range(n_nodes))

        def traverse(n: Node, iy: int, maxD: int, sz: int) -> Node:
            # if we reached the node, apply grow to generate a new subtree     
            if iy == 0:
                return self.tree_random_grow(1, maxD, sz)
            # this should never happen         
            if iy < 0:
                return copy.deepcopy(n)
            children = []
            iy = iy - 1
            maxD = maxD - 1
            sz = sz - 1
            # if we are not there yet, keep applying traverse to each childre adjusting the current maximum depth and size     
            for child in n.children:
                children.append(traverse(child, iy, maxD, sz))
                iy = iy - node_size(children[-1])
                maxD = maxD - 1
                sz = sz - node_size(children[-1])
            return Node(n.function, children)
        # run traverse until it reaches the ix-th node 
        return traverse(n, ix, max_depth, size)

    def expression(self, genome: list) -> list[str]:
        '''
        Convert a tree into string format.

        :return: a list of `str` for each tree in the multi-tree representation.
        '''

        def print_node(node: Node):
            # if the node is terminal, just print the value enclosed in "()"     
            if len(node.children) == 0:
                return node.function.name + "(" + str(node.function()) + ")"
            else:
                # otherwise, call print_node recursively for each children and 
                # concatenate the strings     
                args = [print_node(child) for child in node.children]
                return node.function.name + "(" + ", ".join(args) + ")"

        return [print_node(g) for g in genome]

    def print_population(self):
        """
        Prints the entire population
        """
        for individual in self.population:
            self.print_individual(individual)

    def print_individual(self, individual):
        """
        Prints information about a single individual.
        """
        print("Genome: " + ";".join(self.expression(individual[0])) + " : Fitness: " + str(individual[1]))

    def evolve(self):
        """
        Runs the evolution steps.

        TODO: implement this in the base class as a default for every algorithm.
        """
        # measure the current time     
        t0 = time.time()
        elapsed = 0
        terminate = False
        best_fitness_job = None
        # for each job, if running parallel executions     
        for job in range(self.config.num_jobs):
            best_fitness = None
            # run for a maximum of generations     
            for generation in range(self.config.max_generations):
                # breed     
                self.breed()
                # evaluate the new population
                best_fitness = self.evaluate()
                # `report_generation` will handle the reporting of every generation according to the config     
                self.report_generation(silent_algorithm= self.config.silent_algorithm,
                                       generation=generation,
                                       best_fitness=best_fitness,
                                       report_interval=self.config.report_interval)
                t1 = time.time()
                delta = t1-t0
                t0 = t1
                elapsed += delta
                # if elapsed time is larger than the maximum time, terminate 
                if elapsed + delta >= self.config.max_time:
                    terminate = True
                    break
                # if we found the ideal fitness, terminate         
                elif self.problem.is_ideal(best_fitness):
                    terminate = True
                    break
            # update the current best between runs and report             
            if best_fitness_job is None or self.problem.is_better(best_fitness, best_fitness_job):
                    best_fitness_job = best_fitness
            self.report_job(job = job,
                            num_evaluations=self.num_evaluations,
                            best_fitness=best_fitness_job,
                            silent_evolver=self.config.silent_evolver,
                            minimalistic_output=self.config.minimalistic_output)
            if terminate:
                break
        return self.best_individual.genome
