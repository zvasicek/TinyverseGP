from src.gp.problem import *
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.llm.tiny_llm import *


hyperparameters = LLMHyperparameters(
    model_id = "Qwen/Qwen2.5-Coder-1.5B", #"deepseek-ai/deepseek-coder-6.7b-instruct", 
    temperature = 0.8,
    max_new_tokens = 1000,
    input_length = 1,
    train_dataset_limit = 15,  # Limit the number of training examples to avoid long prompts
    error_penalty = 0
)

generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n,m])
problem = ProgramSynthesis(benchmark.dataset)

prompt = "Write a Python function named \"calculate\" that takes a single input x and returns 1 if x is a power of 2, and 0 otherwise. Ensure the output is formatted as Python code."

tinyllm = TinyLLM(problem, hyperparameters, prompt)    # Prompt is optional, if not provided, it will be built from the dataset
tinyllm.generate()
