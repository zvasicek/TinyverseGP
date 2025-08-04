"""
This file contains the SRBench class which is used to define the configuration of the symbolic regression benchmarking problem.
"""

from gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQRT, SQR, CUBE
from gp.tinyverse import Const, Var, GPConfig, GPHyperparameters
from gp.tiny_cgp import CGPConfig, CGPHyperparameters, TinyCGP
from gp.tiny_tgp import TinyTGP, TGPHyperparameters, Node
from gp.loss import mean_squared_error, linear_scaling_mse, linear_scaling_coeff
from gp.problem import Problem, BlackBox

import re
from sklearn.base import RegressorMixin
import sympy as sp
import numpy as np

strfun = {
    "+": ADD,
    "-": SUB,
    "*": MUL,
    "/": DIV,
    "exp": EXP,
    "log": LOG,
    "square": SQR,
    "cube": CUBE,
}


class SRBench(RegressorMixin):
    def __init__(
        self,
        representation,
        config,
        hyperparameters,
        functions=["+", "-", "*", "/", "exp", "log", "square", "cube"],
        terminals=[1, 0.5, np.pi, np.sqrt(2)],
        scaling_=False,
    ):
        self.representation = representation
        self.scaling = scaling_
        self.loss = linear_scaling_mse if self.scaling else mean_squared_error
        self.functions = [strfun[f] for f in functions]
        self.locals = {f.name: f.custom for f in self.functions if f.custom is not None}
        self.terminals = [Const(c) for c in terminals]
        self.fitted_ = False
        self.config = config
        self.hyperparameters = hyperparameters

    def fit(self, X, y, checkpoint=None):
        problem = BlackBox(X, y, self.loss, 1e-16, True)
        self.terminals = [Var(i) for i in range(X.shape[1])] + self.terminals
        if self.representation == "TGP":
            self.model = TinyTGP(
                self.functions, self.terminals, self.config, self.hyperparameters
            )
        elif self.representation == "CGP":
            self.model = TinyCGP(
                self.functions, self.terminals, self.config, self.hyperparameters
            )
        else:
            raise ValueError("Invalid representation type")
        if checkpoint is not None:
            self.model.resume(checkpoint, problem)
        self.program = self.model.evolve(problem)
        if self.representation == "TGP" and self.scaling:
            yhat = np.array([self.model.predict(self.program.genome, x)[0] for x in X])
            a, b = linear_scaling_coeff(yhat, y)
            self.program.genome[0] = Node(
                ADD,
                [
                    Node(MUL, [Node(Const(a), []), self.program.genome[0]]),
                    Node(Const(b), []),
                ],
            )
        self.fitted_ = True

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        return np.array([self.model.predict(self.program.genome, x)[0] for x in X])

    def get_model(self, X=None):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        expr = self.model.expression(self.program.genome)[0]
        # replace all occurrences of 'Var(i)' in expr with xi
        if X is None:
            expr = re.sub(r"Var\((\d+)\)", r"x\1", expr)
            expr = re.sub(
                r"Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)", r"\1", expr
            )
            expr = sp.sympify(expr, locals=self.locals)
        else:
            # replace all occurrences of 'Var(i)' in expr with the values in X[i]
            self.locals.update({x: sp.Symbol(x) for x in X})
            expr = re.sub(r"Var\((\d+)\)", lambda m: str(X[int(m.group(1))]), expr)
            expr = re.sub(
                r"Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)", r"\1", expr
            )
            expr = sp.sympify(expr, locals=self.locals)
        return expr
