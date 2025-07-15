"""
This file contains the SRBench class which is used to define the configuration of the symbolic regression benchmarking problem.
"""

from gp.functions import ADD, SUB, MUL, DIV
from gp.tinyverse import Const, Var, GPConfig, GPHyperparameters
from gp.tiny_cgp import CGPConfig, CGPHyperparameters, TinyCGP
from gp.tiny_tgp import TinyTGP
from gp.loss import euclidean_distance
from gp.problem import Problem, BlackBox

import re 
import sympy as sp

strfun = {'+': ADD, '-': SUB, '*': MUL, '/': DIV}

class SRBench():
    def __init__(self, representation='TGP', max_generations=100, max_time=60, pop_size=100, functions=['+','-','*','/'], terminals=[1,2,0.5], mutation_rate=0.2, cx_rate=0.9, tournament_size=2):
        self.representation = representation
        self.loss = euclidean_distance
        self.functions = [strfun[f] for f in functions]
        self.locals = {f.name : f.custom for f in self.functions if f.custom is not None}
        self.terminals = [Const(c) for c in terminals] 
        self.fitted_ = False
        if representation == 'TGP':
            self.config = GPConfig(
                            num_jobs=1,
                            max_generations=max_generations,
                            stopping_criteria=1e-6,
                            minimizing_fitness=True, 
                            ideal_fitness=1e-16,
                            silent_algorithm=True,
                            silent_evolver=True,
                            minimalistic_output=True,
                            num_outputs=1,
                            report_interval=1000000,
                            max_time=max_time
                        )
            self.hyperparameters = GPHyperparameters(
                                    pop_size=pop_size,
                                    max_size=50,
                                    max_depth=5,
                                    cx_rate=cx_rate,
                                    mutation_rate=mutation_rate,
                                    tournament_size=tournament_size
                                )
        elif representation == 'CGP':
            self.config = CGPConfig(
                            num_jobs=1,
                            max_generations=max_generations,
                            stopping_criteria=1e-16,
                            minimizing_fitness=True,
                            ideal_fitness=1e-16,
                            silent_algorithm=True,
                            silent_evolver=True,
                            minimalistic_output=True,
                            num_functions=len(functions),
                            max_arity=2,
                            num_inputs=1,
                            num_outputs=1,
                            num_function_nodes=10,
                            report_interval=100000,
                            max_time=max_time
                        )
            self.config.init()
            self.hyperparameters = CGPHyperparameters(
                                    mu=pop_size,
                                    lmbda=pop_size,
                                    population_size=pop_size,
                                    levels_back=10,
                                    mutation_rate=mutation_rate,
                                    strict_selection=True
                                )
        else:
            raise ValueError('Invalid representation type')
    def fit(self, X, y):
        problem  = BlackBox(X, y, self.loss, 1e-16, True)
        self.terminals = [Var(i) for i in range(X.shape[1])] + self.terminals
        if self.representation == 'TGP':
            self.model = TinyTGP(problem, self.functions, self.terminals, self.config, self.hyperparameters)
        elif self.representation == 'CGP':
            self.model = TinyCGP(problem, self.functions, self.terminals, self.config, self.hyperparameters)
        else:
            raise ValueError('Invalid representation type')
        self.program = self.model.evolve()
        self.fitted_ = True 

    def predict(self, X):
        if not self.fitted_:
            raise ValueError('Model not fitted')
        return np.array([self.model.predict(self.program, x)[0] for x in X])
    def get_model(self, X=None):
        if not self.fitted_:
            raise ValueError('Model not fitted')
        expr = self.model.expression(self.program)[0]
        # replace all occurrences of 'Var(i)' in expr with xi
        if X is None:
            expr = sp.sympify(re.sub(r'Var\((\d+)\)', r'x\1', expr), locals=self.locals)
            expr = re.sub(r'Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)', r"\1", expr)
        else:
            # replace all occurrences of 'Var(i)' in expr with the values in X[i]
            self.locals.update({x:sp.Symbol(x) for x in X})
            expr = re.sub(r'Var\((\d+)\)', lambda m: str(X[int(m.group(1))]), expr)
            expr = re.sub(r'Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)', r"\1", expr)
            expr = sp.sympify(expr, locals=self.locals)
        return expr


