"""
Example module to test TinyverseGP on SRBench.

More information about SRBench can be obtained here:

https://cavalab.org/srbench/
https://github.com/cavalab/srbench/tree/master
"""

from benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gp.tiny_cgp import CGPConfig, CGPHyperparameters
from gp.tiny_tgp import TGPHyperparameters, TGPConfig

MAXTIME = 3600  # 1 hour
MAXGEN = 100
POPSIZE = 100
group_datasets = [
    ["522_pm10", "678_visualizing_environmental", "192_vineyard", "1028_SWD"],
    ["1199_BNG_echoMonths", "210_cloud", "1089_USCrime", "1193_BNG_lowbwt"],
    [
        "557_analcatdata_apnea1",
        "650_fri_c0_500_50",
        "579_fri_c0_250_5",
        "606_fri_c2_1000_10",
    ],
]

functions = ["+", "-", "*", "/", "exp", "log", "square", "cube"]
terminals = [1, 0.5, np.pi, np.sqrt(2)]

# Set up hyperparameters for TGP and CGP
tgp_hyperparams = TGPHyperparameters(
    max_depth=8,
    max_size=100,
    pop_size=POPSIZE,
    tournament_size=3,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc=False,  # ephemeral random constants
)
cgp_hyperparams = CGPHyperparameters(
    mu=2,
    lmbda=10,
    strict_selection=True,
    mutation_rate=0.3,
    population_size=POPSIZE,
    levels_back=10,
)

#   Set up configurations for TGP and CGP
tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-16,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000000,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_tgp",
)
cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
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
    num_function_nodes=30,
    report_interval=10,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_cgp",
)
# cgp_config.init()

for g in group_datasets:
    for d in g:
        print(f"Running dataset: {d}\n")
        X, y = fetch_data(d, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        tgp = SRBench(
            "TGP",
            tgp_config,
            tgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=True,
        )
        cgp = SRBench(
            "CGP",
            cgp_config,
            cgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False,
        )

        cgp.fit(
            train_X, train_y
        )  # , checkpoint="examples/checkpoint/srbench_cgp/checkpoint_gen_40.dill")
        print(cgp.get_model())
        print(f"cgp train score: {cgp.score(train_X, train_y)}")
        print(f"cgp test score: {cgp.score(test_X, test_y)}")
        tgp.fit(train_X, train_y)
        print(tgp.get_model())
        print(f"tgp train score: {tgp.score(train_X, train_y)}")
        print(f"tgp test score: {tgp.score(test_X, test_y)}")
        print("=" * 50)
