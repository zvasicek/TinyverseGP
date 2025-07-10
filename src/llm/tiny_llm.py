import random
import re
import textwrap
from .llm import LLMInterface
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from dataclasses import dataclass
from types import FunctionType
from typing import Callable
from src.gp.problem import *


logging.set_verbosity_error()  

@dataclass
class LLMHyperparameters():
    model_id: str
    temperature: float
    max_new_tokens: int
    input_length: int
    train_dataset_limit: int
    error_penalty: float

class TinyLLM(LLMInterface):

    def __init__(self, problem_: object, hyperparameters_: LLMHyperparameters, prompt_: str = None):
        self.problem = problem_ 
        self.hyperparameters = hyperparameters_
        self.prompt = prompt_ 
        self.generated_function = None

    def evaluate(self) -> float:
        try:
            f = self.problem.evaluate(self.generated_function, self)
            return f
        except Exception as e:
            # print(f"Error during evaluation: {e}")
            return self.hyperparameters.error_penalty

    # TODO: Suppress prints from the generated code
    # TODO: Add a timeout for the execution of the generated code
    def predict(self, model:Callable, observation: list) -> list:
        inputs = observation[:self.hyperparameters.input_length]
        return [model(*inputs)]

    def build_prompt(self) -> str:
        if self.prompt is not None:
            return self.prompt.strip() + "\n\n"

        def format_dataset(dataset, input_length):
            random.shuffle(dataset)
            lines = []
            i = 0
            for tup in dataset:
                inputs = ", ".join(map(str, tup[:input_length]))
                outputs = ", ".join(map(str, tup[input_length:]))
                lines.append(f"{inputs} -> {outputs}")
                i += 1
                if i >= self.hyperparameters.train_dataset_limit:
                    break
            return "\n".join(lines)

        prompt = "Given the input-output pairs:\n\n"      
        prompt += format_dataset(self.problem.dataset, self.hyperparameters.input_length)
        prompt += "\n\nWrite a simple function named \"calculate\" for solving above input-output pairs that generalizes also to unseen data. " \
            + "Use only simple Python code without any imports and return only the requested function. Mark the code output as Python code."

        return prompt

    def extract_result(self, text: str) -> tuple:
        try:
            match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                match_plain = re.search(
                    r"(def\s+\w+\s*\(.*?\):(?:\n[ \t]+.*)+)", text, re.DOTALL
                )
                if not match_plain:
                    return None, None
                code = match_plain.group(1)

            code = textwrap.dedent(code)

            namespace = {}
            exec(code, namespace, namespace)

            functions = {name: obj for name, obj in namespace.items() if isinstance(obj, FunctionType)}
            if not functions:
                return None, None
            
            return code, next(iter(functions.values()))
        except Exception as e:
            # print(f"Error extracting result: {e}")
            return None, None

    def invoke_llm(self, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.hyperparameters.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.hyperparameters.model_id).to("cuda") #.to("cpu")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.hyperparameters.max_new_tokens,
            temperature=self.hyperparameters.temperature,
            pad_token_id=tokenizer.pad_token_id
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        return llm.invoke(prompt)

    def generate(self) -> tuple:
        prompt = self.build_prompt()
        response = self.invoke_llm(prompt)
        #print(response)

        str_code, callable_code = self.extract_result(response)
        self.generated_function = callable_code

        fitness = self.evaluate()

        print(f">>> Fitness: {fitness}")

        return str_code, callable_code
