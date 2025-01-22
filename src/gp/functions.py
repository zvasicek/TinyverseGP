from src.tinyverse import Function
import operator

def pdiv(x, y):
    return x / y if y > 0 else 1.0

# Arithmetic Functions
ADD = Function(2, 'ADD', operator.add)
SUB = Function(2, 'SUB', operator.sub)
MUL = Function(2, 'MUL', operator.mul)
DIV = Function(2, 'DIV', pdiv)

# Logical Functions
AND = Function(2, 'AND', operator.and_)
OR  = Function(2, 'OR', operator.or_)
NAND = Function(2, 'NAND', operator.not_(operator.and_))
NOR = Function(2, 'Nor', operator.not_(operator.or_))
NOT = Function(1, 'NOR', operator.not_)

# Policy Search
