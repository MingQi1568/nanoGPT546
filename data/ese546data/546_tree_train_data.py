"""
Data Class for generating training data

RESOLVED:
- redundant parentheses (will not have these)
- multiple solving strategies (only accept post order traversal)
- negatives and floats (fixed precision decided)
- rebalancing for guaranteed post-order traversal

TODO: 
- 
"""


import random
import math
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import os

# --- 1. Abstract Base Classes ---

class Node(ABC):
    @property
    @abstractmethod
    def precedence(self) -> int:
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def render(self, mask_node: Optional['Node'] = None) -> str:
        pass

class Number(Node):
    precedence = 10 

    def __init__(self, value: Union[int, float], precision: int = 4):
        self.value = value
        self.precision = precision

    def evaluate(self) -> float:
        return self.value

    def render(self, mask_node: Optional['Node'] = None) -> str:
        return f"{self.value:.{self.precision}f}"

class Operation(Node):
    def __init__(self, left: Node, right: Optional[Node] = None):
        self.left = left
        self.right = right 

    @property
    @abstractmethod
    def tool_name(self) -> str:
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    @property
    def is_unary(self) -> bool:
        """Helper to identify if this is a function like sin/abs"""
        return self.right is None

    def get_tool_signature(self, precision: int) -> str:
        op_vals = [self.left.evaluate()]
        if self.right:
            op_vals.append(self.right.evaluate())
        
        vals_str = " ".join([
            f"{v:.{precision}f}"
            for v in op_vals
        ])
        return f"|\\{self.tool_name} {vals_str} "  # Added trailing space

    def _should_wrap(self, child: Node, is_right_side: bool = False) -> bool:
        """
        Determines if a child node needs parentheses.
        """
        if self.is_unary:
            return False

        # 1. If child is weaker, wrap it.
        if child.precedence < self.precedence:
            return True
        
        # 2. Ambiguity check for equal precedence
        if child.precedence == self.precedence and is_right_side:
            if self.tool_name in ['sub', 'div']: # NON_ASSOCIATIVE OPS
                return True
                
        return False

    def render(self, mask_node: Optional['Node'] = None) -> str:
        if self is mask_node:
            return "$"

        # --- LEFT SIDE ---
        l_str = self.left.render(mask_node)
        
        # Unary check is handled inside _should_wrap now or explicitly here
        if self.left is not mask_node and self._should_wrap(self.left, is_right_side=False):
            l_str = f"({l_str})"

        # --- RIGHT SIDE (Binary) ---
        if self.right:
            r_str = self.right.render(mask_node)
            
            if self.right is not mask_node and self._should_wrap(self.right, is_right_side=True):
                r_str = f"({r_str})"
            
            return f"{l_str}{self.symbol}{r_str}"
        
        # --- UNARY SIDE ---
        return f"{self.symbol}({l_str})"


# --- 2. Concrete Operations ---

class Add(Operation):
    tool_name = "add"
    symbol = "+"
    precedence = 1
    def evaluate(self): return self.left.evaluate() + self.right.evaluate()

class Sub(Operation):
    tool_name = "sub"
    symbol = "-"
    precedence = 1
    def evaluate(self): return self.left.evaluate() - self.right.evaluate()

class Mul(Operation):
    tool_name = "mul"
    symbol = "*"
    precedence = 2
    def evaluate(self): return self.left.evaluate() * self.right.evaluate()

class Div(Operation):
    tool_name = "div"
    symbol = "/"
    precedence = 2
    def evaluate(self): 
        denom = self.right.evaluate()
        return self.left.evaluate() / denom if denom != 0 else float('nan')
    
class Abs(Operation):
    tool_name = "abs"
    symbol = "abs"
    precedence = 3
    def evaluate(self): return abs(self.left.evaluate())

class Sin(Operation):
    tool_name = "sin"
    symbol = "sin"
    precedence = 3 
    def evaluate(self): return math.sin(self.left.evaluate())

class Cos(Operation):
    tool_name = "cos"
    symbol = "cos"
    precedence = 3
    def evaluate(self): return math.cos(self.left.evaluate())
    
class Log(Operation):
    """Assumed Base e"""
    tool_name = "log"
    symbol = "log"
    precedence = 3 
    def evaluate(self): 
        val = self.left.evaluate()
        if val <= 0: return float("nan")
        return math.log(val)

class Exp(Operation):
    tool_name = "exp"
    symbol = "exp"
    precedence = 3
    def evaluate(self): 
        val = self.left.evaluate()
        try:
            return math.exp(val)
        except OverflowError:
            return float("inf")

# --- 3. The Generator Engine ---

class MathDataGenerator:
    def __init__(self, max_depth: int = 3, max_val: int = 20, precision: int = 4, init_negative = False, init_float = False, registry: list[list[Operation]] = None):
        self.max_depth = max_depth
        self.max_val = max_val
        self.precision = precision
        self.init_negative = init_negative
        self.init_float = init_float
        if not registry:
            self.binary_ops = [Add, Sub, Mul, Div] # ADD ALL NEW OPERATIONS HERE
            self.unary_ops = [Sin, Cos, Log, Exp, Abs]
        else:
            self.binary_ops = registry[0]
            self.unary_ops = registry[1]
        
    def rebalance_tree(self, node: Node) -> Node:
        """
        Rotates Right-Heavy trees to Left-Heavy trees where mathematically valid.
        
        Valid Rotations:
        1. Add + Add:  A + (B + C) -> (A + B) + C
        2. Mul + Mul:  A * (B * C) -> (A * B) * C
        3. Add + Sub:  A + (B - C) -> (A + B) - C
        4. Mul + Div:  A * (B / C) -> (A * B) / C
        
        Invalid Rotations (Do NOT change):
        1. Sub + Add:  A - (B + C) != (A - B) + C  (Because it distributes to A - B - C)
        2. Div + Mul:  A / (B * C) != (A / B) * C  (Because it distributes to A / B / C)
        """
        if isinstance(node, Number):
            return node

        # 1. Recursively rebalance children first
        if isinstance(node, Operation):
            node.left = self.rebalance_tree(node.left)
            if node.right:
                node.right = self.rebalance_tree(node.right)

        # 2. Check for rotation opportunities
        if isinstance(node, Operation) and node.right and isinstance(node.right, Operation):
            
            # --- LOGIC UPDATE: CHECK FAMILIES ---
            # Is this an associative family? (Add/Sub) or (Mul/Div)
            is_add_fam = isinstance(node, (Add, Sub)) and isinstance(node.right, (Add, Sub))
            is_mul_fam = isinstance(node, (Mul, Div)) and isinstance(node.right, (Mul, Div))
            
            if is_add_fam or is_mul_fam:
                # We can only rotate if the PARENT (node) is commutative-friendly.
                # If Parent is SUB or DIV, we cannot rotate the right child up 
                # without changing the math logic significantly.
                
                can_rotate = True
                if isinstance(node, Sub): can_rotate = False
                if isinstance(node, Div): can_rotate = False
                
                if can_rotate:
                    # Perform Rotation
                    # Current: Node( left=A, right=Pivot(left=B, right=C) )
                    # Target:  Pivot( left=Node(left=A, right=B), right=C )
                    
                    # Be careful: The pivot takes the operator of the RIGHT child, 
                    # but the new inner node takes the operator of the PARENT.
                    # Example: A + (B - C)
                    # Node = Add, Pivot = Sub
                    # Result should be Sub( Add(A, B), C ) -> (A+B)-C
                    
                    pivot = node.right
                    
                    # 1. Create the new inner node (A ? B)
                    # It copies the current Node's class (Add or Mul)
                    new_left = type(node)(node.left, pivot.left)
                    
                    # 2. The Node becomes the Pivot's class (Sub or Div or Add or Mul)
                    # We have to mutate 'node' in place to keep the tree reference valid, 
                    # OR return a new node. Since we are recursive, returning new node is safer.
                    
                    # Create the new root (The Sub or Div)
                    new_root = type(pivot)(new_left, pivot.right)
                    
                    # Recursively check again in case we pulled up another rotatable node
                    return self.rebalance_tree(new_root)
        
        return node

    def generate_random_tree(self, current_depth: int = 0) -> Node:
        if current_depth >= self.max_depth or (current_depth > 0 and random.random() < 0.3):
            match (self.init_float, self.init_negative):
                case (False, False):
                    val = random.randint(1, self.max_val)
                case (False, True):
                    val = random.randint(-self.max_val, self.max_val)
                case (True, False):
                    val = random.uniform(0, self.max_val)
                case (True, True):
                    val = random.uniform(-self.max_val, self.max_val)
            return Number(val, self.precision)
        
        if random.random() < 0.8:
            op_class = random.choice(self.binary_ops)
            left = self.generate_random_tree(current_depth + 1)
            right = self.generate_random_tree(current_depth + 1)
            # Rebalance immediately upon creation
            return self.rebalance_tree(op_class(left, right))
        else:
            op_class = random.choice(self.unary_ops)
            child = self.generate_random_tree(current_depth + 1)
            return op_class(child, None)

    def solve_step_by_step(self, root: Node) -> str:
        lines = []
        while True:
            lines.append(root.render())
            target = self._find_next_op(root)
            if not target: break
            context = root.render(mask_node=target)
            tool_sig = target.get_tool_signature(self.precision)
            lines.append(f"{context}{tool_sig}")
            result_val = target.evaluate()
            # Standard output format for chain of thought
            baked_val = float(f"{result_val:.{self.precision}f}")
            if baked_val == 0: baked_val = 0.0 # Standardize negative zero
            lines.append(f"={baked_val:.{self.precision}f}")
            new_node = Number(baked_val, self.precision)
            if root is target:
                root = new_node
            else:
                self._replace_child(root, target, new_node)
        lines.append(f"{root.evaluate():.{self.precision}f}\\&\\")
        return "\n".join(lines)

    def _find_next_op(self, node: Node) -> Optional[Operation]:
        if isinstance(node, Number): return None
        if isinstance(node, Operation):
            l_found = self._find_next_op(node.left)
            if l_found: return l_found
            if node.right:
                r_found = self._find_next_op(node.right)
                if r_found: return r_found
            left_ready = isinstance(node.left, Number)
            right_ready = isinstance(node.right, Number) if node.right else True
            if left_ready and right_ready:
                return node
        return None

    def _replace_child(self, parent: Node, old_child: Node, new_child: Node):
        if isinstance(parent, Operation):
            if parent.left is old_child: parent.left = new_child
            elif parent.right is old_child: parent.right = new_child
            else:
                self._replace_child(parent.left, old_child, new_child)
                if parent.right: self._replace_child(parent.right, old_child, new_child)

# --- 4. Execution ---

def main():
    gen = MathDataGenerator(max_depth=4, max_val=10)

    print("--- 1. Rebalance Test: A + (B - C) ---")
    # Should become (A + B) - C
    # Solver should do A+B first, then result - C
    tree = Add(Number(5), Sub(Number(10), Number(3))) 
    print(f"Original: 5 + (10 - 3)")
    balanced = gen.rebalance_tree(tree)
    print(gen.solve_step_by_step(balanced))
    
    print("\n--- 2. NON-Rebalance Test: A - (B + C) ---")
    # Should STAY A - (B + C) because subtraction is not associative
    tree = Sub(Number(20), Add(Number(5), Number(3)))
    print(f"Original: 20 - (5 + 3)")
    balanced = gen.rebalance_tree(tree)
    print(gen.solve_step_by_step(balanced))

    print("\n--- 3. Unary Parenthesis Check ---")
    # Should render sin(5+3) not sin((5+3))
    # Unary ops generally handle their own brackets
    tree = Sin(Add(Number(5), Number(3)))
    print(gen.solve_step_by_step(tree))
    
    print("\n--- Test for early rounding ---")
    tree = Mul(Div(Number(10), Number(3)), Number(3))
    print(gen.solve_step_by_step(tree))
    
    print("\n--- 4. Complex Random ---")
    rand = gen.generate_random_tree()
    print(gen.solve_step_by_step(rand))
    
def dataset_1(filepath, count=10_000):
    # depth = 2

    registry = [
        [Add, Sub, Mul],
        [Abs]
    ]

    gen = MathDataGenerator(max_depth=2, precision = 0, registry=registry, init_negative = False)
    
    input_file_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(input_file_path, "w") as f:
        for i in range(count):
            rand = gen.generate_random_tree()
            result = gen.solve_step_by_step(rand)
            f.write(str(result))
            f.write("\n")
    
def dataset_2(filepath, count=10_000, precision=4):
    # dataset_1 AND
    # /, sin, cos, ln, exp
    # depth = 2, floats too
    # initialize negative
    registry = [
        [Add, Sub, Mul, Div],
        [Abs, Sin, Cos, Log, Exp]
    ]
    gen = MathDataGenerator(max_depth=2, precision=precision, init_negative=True, registry=registry)

    input_file_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(input_file_path, "w") as f:
        for i in range(count):
            rand = gen.generate_random_tree()
            result = gen.solve_step_by_step(rand)
            f.write(str(result))
            f.write("\n")
            
def dataset_train(filepath, count=10_000, precision=4):
    registry = [
        [Add, Sub, Mul, Div],
        [Abs, Sin, Cos, Log, Exp]
    ]
    gen = MathDataGenerator(max_depth=2, precision=precision, init_negative=True, init_float=True, registry=registry, max_val=20)

    input_file_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(input_file_path, "w") as f:
        for i in range(count):
            rand = gen.generate_random_tree()
            result = gen.solve_step_by_step(rand)
            f.write(str(result))
            f.write("\n")
            
def dataset_test(filepath, count=10_000, precision=4):
    registry = [
        [Add, Sub, Mul, Div],
        [Abs, Sin, Cos, Log, Exp]
    ]
    gen = MathDataGenerator(max_depth=2, precision=precision, init_negative=True, init_float=True, registry=registry, max_val=100_000)

    input_file_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(input_file_path, "w") as f:
        for i in range(count):
            rand = gen.generate_random_tree()
            result = gen.solve_step_by_step(rand)
            f.write(str(result))
            f.write("\n")
    
    
def dataset_3(filepath):
    # dataset_2 AND
    # higher depth
    # initialize with floats
    gen = MathDataGenerator(max_depth=3, max_val=50, precision=4, init_negative=True, init_float=True)
    
    for i in range(10):
        print(f"\nProblem {i+1}: ")
        rand = gen.generate_random_tree()
        print(gen.solve_step_by_step(rand))


if __name__ == "__main__":
    # main()
    # dataset_1()
    dataset_train("training.txt", count=300_000, precision=4)
    dataset_test("testing.txt", count=1_000, precision=4)
    # dataset_3()