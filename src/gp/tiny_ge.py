"""
TinyGP: A minimalistic implementation of Grammatical Evolution GE for
         TinyverseGP.
"""

import random
import copy
import time
import re
from src.gp.problem import *
from src.gp.tinyverse import *


@dataclass
class GEHyperparameters(Hyperparameters):
    """
    Specialized hyperparameter configuration space for GE.
    """
    pop_size: int
    genome_length: int
    codon_size: int
    cx_rate: float
    mutation_rate: float
    tournament_size: int

class TinyGE(GPModel):
    '''
    Main class of the tiny GE module that derives from GPModel and
    implements all related fundamental mechanisms to run GE.
    '''
    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: list[Function]

    def __init__(self, problem_: object, functions_: list[Function], grammar_: dict, arguments_: list[str], config: Config, hyperparameters: Hyperparameters):
        self.functions = {f.name.upper(): f.function for f in functions_} # the list of functions to that could be used in the grammar                                 # TODO: Adjust to updates in the framework
        self.grammar = grammar_ # the defined grammar
        self.arguments = arguments_ # the arguments for the functions to be generated
        self.problem = problem_ # an instance to a problem. This allows us to handle different problems transparently
        self.hyperparameters = hyperparameters # hyperparameters
        self.config = config # overall configuration
        self.best = None  # to keep the best program found so far
        self.num_evaluations = 0 # counter of number of evaluations

        # initial population using uniform initialization
        self.population = [[genome, 0.0] for genome in self.init_uniform(self.hyperparameters.pop_size, self.hyperparameters.genome_length, self.hyperparameters.codon_size)]        

        self.evaluate() # evaluates the initial population

    def init_uniform(self, num_pop: int, max_genome_length: int, codon_size: int):
        '''
        Initialize the population uniformly. It will create one genome per output.

        :return: a list of lists
        '''
        pop = []
        for _ in range(num_pop):
            pop.append([random.randint(0, codon_size) for _ in range(max_genome_length)])
        return pop

    def evaluate(self) -> float:
        '''
        Triggers the evaluation of the whole population.

        :return: a `float` value of the best fitness
        '''
        best = None
        # For each individual in the population 
        for ix, individual in enumerate(self.population):
            genome = individual[0]  # extract the genome
            fitness = self.evaluate_individual(genome) # evaluate it
            self.population[ix][1] = fitness # assign the fitness
            # update the population best solution
            if best is None or self.problem.is_better(fitness, best):
                best = fitness
            # update the best solution of all time
            if self.best is None or self.problem.is_better(fitness, self.best[1]):
                self.best = [copy.copy(genome), fitness]
        return best

    def evaluate_individual(self, genome:list[int]) -> float:
        '''
        Evaluate a single individual `genome`.

        :return: a `float` representing the fitness of that individual.
        '''
        self.num_evaluations += 1  # update the evaluation counter
        f = self.problem.evaluate(genome, self) # evaluate the solution using the problem instance
        if self.best is None or self.problem.is_better(f, self.best[1]):
            self.best = [copy.copy(genome), f]
        return f

    def predict(self, genome: list, observation: list) -> list:
        '''
        Predict the output of the `genome` given a single `observation`.

        :return: a list of the outputs for that observation
        '''
        def evaluate_expression(expr: str, func_dict: list, args: list[str], values: list) -> any:
            local_vars = dict(zip(args, values))
            return [eval(expr, func_dict, local_vars)]

        tmp_expr = self.expression(genome)
        if '<' in tmp_expr or '>' in tmp_expr:
            #return [float('nan')]                                                                # TODO: CHANGE THIS!!! / Penalty for invalid genome? --> problem.py line 109
            return [999999999999999] 
        
        return evaluate_expression(tmp_expr, self.functions, self.arguments, observation)

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
        self.population.append([copy.copy(self.best[0]), self.best[1]])

    def perturb(self, parent1: list, parent2: list) -> list:
        '''
        Applies the crossover and mutation operators to the parents.

        :return: a list of the `genome` and `None` representing the unevaluated fitness.
        '''
        # applies the crossover with `self.hyperparameters.cx_rate` probability, otherwise return the first parent
        genome = self.crossover(parent1, parent2) if random.random() <= self.hyperparameters.cx_rate else parent1
        # applies mutation with `self.hyperparameters.mutation_rate`
        genome = self.mutation(genome, 0, self.hyperparameters.codon_size, self.hyperparameters.mutation_rate)
        return [genome, None] # returns the unevaluated offspring

    def selection(self) -> list:            
        '''
        Select a parent from the population using the tournament selection method.

        :return: a selected genome.
        '''
        # samples `self.hyperparameters.tournament_size` solutions completely at random
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        # return the best of this sample whether it is a minimization or maximization problem     
        if self.problem.minimizing:
            return min(parents, key=lambda ind: ind[1])[0]
        else:
            return max(parents, key=lambda ind: ind[1])[0]

    def crossover(self, p1: list, p2: list) -> list:
        '''
        Applies onepoint crossover to the given parents.

        :return: the recombined genomes.
        '''   
        parent1 = copy.deepcopy(p1)
        parent2 = copy.deepcopy(p2)

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return random.choice([child1, child2])


    def mutation(self, genome: list, min_val: int, max_val: int, mutation_rate: float) -> list:
        '''
        Apply the int flip per codon mutation to the parent.

        :return: the mutated genome.
        '''
        mutated = copy.deepcopy(genome)

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                old_val = mutated[i]
                possible_values = list(range(min_val, max_val + 1))
                possible_values.remove(old_val)
                mutated[i] = random.choice(possible_values) if possible_values else old_val

        return mutated

    def genotype_phenotype_mapping(self, grammar, genome, expression='<expr>'):
        '''
            Maps the genotype to its phenotype.
            
            :return: a string representation of the genome.
        '''
        tmp_genome = copy.deepcopy(genome)
        while '<' in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r'<(.*?)>', expression).group(0)
            choice = grammar[next_non_terminal][(tmp_genome.pop(0) % len(grammar[next_non_terminal]))]
            expression = expression.replace(next_non_terminal, choice, 1)
        return expression

    def expression(self, genome: list) -> str:
        '''
        Convert a genome into string format with the help of the grammar.

        :return: expression as `str`.
        '''
        return self.genotype_phenotype_mapping(self.grammar, genome, '<expr>')

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
        print("Expression: " + ";".join(self.expression(individual[0])) + " : Fitness: " + str(individual[1]))

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
                self.report_generation(silent = self.config.silent_algorithm,
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
        return self.best[0]
