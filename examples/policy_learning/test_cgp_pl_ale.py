"""
Example module to test CGP with policy search problems.
Evolves a policy for Pong from the Gymnasium Atari Learning Environment:

https://ale.farama.org/
https://ale.farama.org/environments/

https://ale.farama.org/environments/pong/

Pong has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(6)

Observation space: Box(0, 255, (210, 160, 3), uint8)
"""

from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs
from src.gp.tiny_cgp import *
import gymnasium as gym
from src.gp.problem import PolicySearch
from src.gp.functions import *
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

ale_args = ALEArgs(
    noop_max=30,
    frame_skip=4,
    screen_size=32,
    grayscale_obs=True,
    terminal_on_life_loss=True,
    scale_obs=False,
    frame_stack=4,
)

env = gym.make("ALE/Pong-v5", frameskip=1, difficulty=0)
benchmark = PLBenchmark(env, ale_=True, ale_args=ale_args, flatten_obs_=False)
wrapped_env = benchmark.wrapped_env
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]
terminals = benchmark.gen_terminals()
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

config = CGPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    num_function_nodes=100,
    report_interval=1,
    max_time=99999,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='pl_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    levels_back=100,
    mutation_rate=0.05,
    strict_selection=True,
)

problem = PolicySearch(env=wrapped_env, ideal_=100, minimizing_=False)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.make("ALE/Pong-v5", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
