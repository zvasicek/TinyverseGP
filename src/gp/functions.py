from src.tinyverse import Function
import operator

def f2b(input: float):
    return True if input > 0 else False

def b2f(input: bool):
    return 1.0 if input else -1.0

def pdiv(x, y):
    return x / y if y > 0 else 1.0

# Arithmetic Functions
ADD = Function(2, 'ADD', operator.add)
SUB = Function(2, 'SUB', operator.sub)
MUL = Function(2, 'MUL', operator.mul)
DIV = Function(2, 'DIV', pdiv)

# Logical Functions
AND = Function(2, 'AND', lambda x,y : int(x) & int(y))
OR  = Function(2, 'OR', lambda x,y : int(x) | int(y))
NOT = Function(1, 'NOT', lambda x : ~int(x))
NAND = Function(2, 'NAND', lambda x,y : ~(int(x) & int(y)))
NOR = Function(2, 'NOR', lambda x,y : ~(int(x) | int(y)))

# Policy Search / Classificiation

LT = Function(2, 'AND', operator.lt)
LTE = Function(2, 'AND', operator.le)
GT = Function(2, 'AND', operator.gt)
GTE = Function(2, 'AND', operator.ge)
EQ = Function(2, 'AND', operator.eq)




