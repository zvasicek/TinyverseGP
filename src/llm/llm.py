from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def build_prompt(self):
        pass

    def extract_result(self):
        pass
