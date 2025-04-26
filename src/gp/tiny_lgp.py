"""
TinyLGP: A minimalistic implementation of Linear Genetic Programming for
         TinyverseGP.

         Genome representation: 
         Mutation operator: 
         Search algorithm: 
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from src.gp.tinyverse import GPModel, Hyperparameters, GPConfig, Var
from src.gp.problem import Problem

