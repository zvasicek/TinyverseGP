from src.gp.tinyverse import GPModel
import copy

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score




class Hpo:
    
    def optimise(gpmodel_ : GPModel):
        """_summary_

        Args:
            gpmodel_ (GPModel): the model to be optimised

        Returns:
            GPHyperparameters: the configuration to be used
        """
        # WARNING : This configuration space is only to test with tinyCGP
        # TODO : include the actual config spaces
        configspace = ConfigurationSpace({"mu": (0.2, 1.0),
                                  "lmbda": (10, 50),
                                  "population_size": (20, 500),
                                  "levels_back": (1, 3),
                                  "strict_selection": [True,False],
                                  "mutation_rate": (0.01, 0.5),
                                  "mutation_rate_genes": (0.01, 0.5),
                                  })
        
        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters,c,config[c])
            gpmodel.evolve()
            return gpmodel.best_individual.fitness
        
        # Scenario object specifying the optimization environment
        scenario = Scenario(configspace, deterministic=True, n_trials=10)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        incHP = copy.deepcopy(gpmodel_.hyperparameters)
        for c in  incumbent.keys():
            setattr(incHP,c,incumbent[c])
        
        return incHP



    