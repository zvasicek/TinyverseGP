import random
import re
import textwrap
import traceback
import contextlib
import os
import time
from .llm import LLMInterface
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from dataclasses import dataclass
from types import FunctionType
from typing import Callable
from src.gp.problem import *
from openai import OpenAI
from multiprocessing import Process, Queue

logging.set_verbosity_error()  

@dataclass
class LLMHyperparameters():
    model_id: str
    temperature: float
    max_new_tokens: int
    input_length: int
    train_dataset_limit: int
    error_penalty: float
    openai_api_key: str    # Optional, if not provided, the defined local model will be used
    timeout: int 
    max_time: int
    max_iterations: int
    minimizing_fitness: bool

def evaluate_worker(queue, problem, func, context):    # To prevent potential endless loops
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            result = problem.evaluate(func, context)
        queue.put(result)
    except Exception as e:
        queue.put(None)

class TinyLLM(LLMInterface):

    def __init__(self, problem_: object, hyperparameters_: LLMHyperparameters, prompt_: str = None):
        self.problem = problem_ 
        self.hyperparameters = hyperparameters_
        self.prompt = prompt_ 
        self.generated_function = None
        self.best_fitness = None
        self.best_str_code = None
        self.best_callable_code = None
        self.tokenizer_store = None
        self.model_store = None

    def evaluate(self) -> float:
        queue = Queue()
        p = Process(target=evaluate_worker, args=(queue, self.problem, self.generated_function, self))
        p.start()
        p.join(timeout=self.hyperparameters.timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            return self.hyperparameters.error_penalty
        else:
            result = queue.get()
            return result if result is not None else self.hyperparameters.error_penalty

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

    def invoke_lokal_llm(self, prompt: str) -> str:
        if self.tokenizer_store is not None and self.model_store is not None:
            tokenizer = self.tokenizer_store
            model = self.model_store
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.hyperparameters.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.hyperparameters.model_id).to("cuda") #.to("cpu")
            self.tokenizer_store = tokenizer
            self.model_store = model

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

    def invoke_openai_llm(self, prompt: str) -> str:
        client = OpenAI(api_key=self.hyperparameters.openai_api_key)

        response = client.chat.completions.create(
            model=self.hyperparameters.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.hyperparameters.temperature
        )
        return response.choices[0].message.content

    def generate(self) -> tuple:
        prompt = self.build_prompt()

        t0 = time.time()
        elapsed = 0
        for _ in range(self.hyperparameters.max_iterations):
            if self.hyperparameters.openai_api_key.strip() not in [None, '']:
                response = self.invoke_openai_llm(prompt)
            else:
                response = self.invoke_lokal_llm(prompt)

            str_code, callable_code = self.extract_result(response)
            self.generated_function = callable_code

            fitness = self.evaluate()
            if self.best_fitness is None or (self.hyperparameters.minimizing_fitness and fitness < self.best_fitness) or (not self.hyperparameters.minimizing_fitness and fitness > self.best_fitness):
                self.best_fitness = fitness
                self.best_str_code = str_code
                self.best_callable_code = callable_code

            print(f">>> Fitness: {self.best_fitness}")

            t1 = time.time()
            delta = t1-t0
            t0 = t1
            elapsed += delta
            if elapsed + delta >= self.hyperparameters.max_time:
                break

        return self.best_str_code, self.best_callable_code
