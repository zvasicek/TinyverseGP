# https://leetcode.com/problems/power-of-two/

import random
import math

N = 100
MAX = 10000

# https://github.com/hongxiaolong/leetcode/blob/master/Python/power_of_two.py
def isPowerOfTwo(n):
    return n > 0 and bin(n).count('1') == 1

leetcode_testcases = [(0, 0), (1, 1), (-1, 0), (16, 1), (65536, 1)]

def generate_dataset(n, max = 10000, ratio = 0.2):
    """
    Generates a dataset of n observations with maximum value max and ratio positive examples

    :param n: Number of observations
    :param max: Maximum value in the dataset
    :param ratio: Ratio of positive examples (leads to 1.0 - ratio counterexamples)
    :return: dataset
    """
    dataset = []
    exp = int(math.log2(max))
    for i in range(n):
        if random.random() < ratio:
            rand =  int(math.pow(2, random.randint(0, exp)))
            status = 1
        else:
            rand = random.randint(0, max)
            status = 1 if isPowerOfTwo(rand) else 0
        case = (rand,  status)
        dataset.append(case)
    return dataset
