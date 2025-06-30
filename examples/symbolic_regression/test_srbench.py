"""
Example module to test TinyverseGP on SRBench.

More information about SRBench can be obtained here:

https://cavalab.org/srbench/
https://github.com/cavalab/srbench/tree/master
"""

from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = fetch_data('cloud', return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

tgp = SRBench('TGP')
cgp = SRBench('CGP')

cgp.fit(train_X, train_y)
print(cgp.get_model())
print(f"cgp test score: {cgp.predict(train_X)}")

tgp.fit(train_X, train_y)
print(tgp.get_model())
print(f"tgp test score: {tgp.score(test_X, test_y)}")

