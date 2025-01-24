from src.gp.tinyverse import Function
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

# Policy Search / Classification

LT = Function(2, 'AND', lambda x,y : b2f(x < y))
LTE = Function(2, 'AND', lambda x,y : b2f(x <= y))
GT = Function(2, 'AND', lambda x,y : b2f(x > y))
GTE = Function(2, 'AND',lambda x,y : b2f(x >= y))
EQ = Function(2, 'AND', lambda x,y : b2f(x == y))
NEG = Function(1, 'AND', lambda x : operator.neg(x))
IF = Function(3, 'AND', lambda x,y,z : y if f2b(x) else z)
IFLEZ = Function(3, 'AND', lambda x,y,z : y if x <= 0 else z)
IFGTZ = Function(3, 'AND', lambda x,y,z : y if x > 0 else z)



