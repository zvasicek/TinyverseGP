import gymnasium as gym
from dataclasses import dataclass
from abc import ABC
from src.benchmark.policy_search.policy_evaluation import GPAgent
from tinyverse import GPModel

class Problem(ABC):
    ideal: float

    def is_ideal(self, fitness: float) -> bool:
        return fitness == self.ideal

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        return fitness1 < fitness2 if self.minimizing \
            else fitness1 > fitness2

@dataclass
class BlackBoxProblem(Problem):
    observations: list
    actual: list

    def __init__(self, data_: list, actual_: list, loss_: callable,
                 ideal_: float, minimizing_: bool):
        self.data = data_
        self.actual = actual_
        self.loss = loss_
        self.ideal = ideal_
        self.minimizing = minimizing_

    def evaluate(self, prediction: list) -> float:
        return self.loss(self.actual, prediction)

class PolicySearchProblem(Problem):
    agent: GPAgent
    model: GPModel

    def __init__(self, env: gym.Env, ideal_: float, minimizing_: bool, model_: GPModel):
        self.agent = GPAgent(env)
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.model = model_

    def evaluate(self, genome) -> float:
        return self.agent.evaluate_policy(genome, self.model)
