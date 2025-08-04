from abc import ABC, abstractmethod
from gp.tinyverse import GPModel, GPHyperparameters
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
import copy


class HPOInterface(ABC):
    """
    Abstract class that is used to implement methods for running HPO with GP models
    that are provided within TinverseGP.
    """

    @abstractmethod
    def optimise(self, gpmodel_: GPModel, *params):
        pass


class SMACInterface(HPOInterface):

    def optimise(self, gpmodel_: GPModel, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC (https://github.com/automl/SMAC3)

        Args:
            gpmodel_ (GPModel): the GP model to be optimised
            n_trials_: Number of trials in the SMAC optimisation environment

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            gpmodel.evolve()
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        paramspace = gpmodel_.hyperparameters.space

        # Use the HP space to init the configuration space (CS)
        configspace = ConfigurationSpace(paramspace)

        # Scenario object specifying the optimization environment
        scenario = Scenario(configspace, deterministic=True, n_trials=n_trials_)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()

        inc_hp = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])
        return inc_hp


class Hpo:
    """
    Class that provides methods to run HPO for GP models
    that are provided within TinverseGP.
    """

    def optimise_smac(gpmodel_: GPModel, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC.

        Args:
            gpmodel_ (GPModel): the model to be optimised

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            gpmodel.evolve()
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        space = gpmodel_.hyperparameters.space

        # Use the HP space to init the configuration space (CS)
        cs = ConfigurationSpace(space)

        # Scenario object specifying the optimization environment
        scenario = Scenario(cs, deterministic=True, n_trials=n_trials_)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        incHP = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(incHP, c, incumbent[c])
        return incHP
