from src.benchmark.srbench import SRBench
import numpy as np

X = np.random.uniform(-10, 10, (100,2))
y = 2*X[:,0]**2 + X[:,1]

tgp = SRBench('TGP')
cgp = SRBench('CGP')

tgp.fit(X, y)
print(tgp.get_model(X=["alpha","beta"]))
cgp.fit(X,y)
print(cgp.get_model(X=["alpha","beta"]))
