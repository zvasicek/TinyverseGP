from abc import ABC
from abc import abstractmethod

class GPBase(ABC):
    best_fitness: float

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def evolve(self):
        pass

    def selection(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def expression(self):
        pass

    @abstractmethod
    def report_job(self):
        pass

    @abstractmethod
    def report_generation(self):
        pass


