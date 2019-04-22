import ast
import pptree
import sys

class Node:
    def __init__(self, name, children, parent=None):
        self.name = name
        self.children = []
        for child in children:
            if child is not None and child.name != 'None':
                self.children.append(child)
        self.parent = parent
        self.visited = False
        for child in self.children:
            child.parent = self

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def pprint(self):
        pptree.print_tree(self, childattr='children', nameattr='name')
        
    def get_terminals(self, terminals=None):
        if terminals == None:
            terminals = []
        if not self.children:
            terminals.append(self)
        else:
            for child in self.children:
                child.get_terminals(terminals=terminals)
        return terminals

def create_ast_tree(code_string):
    def generate_name_from_node(node):
        name = node.__class__.__name__
        value_generator = {
            'Num': lambda x: str(x.n),
            'Str': lambda x: str(x.s),
            'FormattedValue': lambda x: generate_name_from_node(x.value),
            'Bytes': lambda x: str(x.s),
            'NameConstant': lambda x: str(x.value),
            'Name': lambda x: str(x.id),
            'BinOp': lambda x: x.op.__class__.__name__,
            'BoolOp': lambda x: x.op.__class__.__name__,
            'keyword': lambda x: str(x.arg)
        }
        if name in value_generator:
            return value_generator[name](node)
        else:
            return name

    def generate_children(nodes):
        children = []
        ctx = ['Load', 'Store', 'Del']
        for node in nodes:
            if isinstance(node, list):
                list_nodes = generate_children(node)
                children.extend(list_nodes)
            elif node.__class__.__name__ not in ctx:
                children.append(create_node(node))
        return children
    
    def create_node(ast_node):
        LITERALS = ['Num', 'Str', 'FormattedValue', 'Bytes', 'NameConstant', 'Name']
        if isinstance(ast_node, ast.AST):
            name = generate_name_from_node(ast_node)
            if ast_node.__class__.__name__ == 'BinOp':
                children = generate_children([ast_node.left, ast_node.right])
                return Node(name, children)
            elif ast_node.__class__.__name__ == 'BoolOp':
                children = generate_children(ast_node.values)
                return Node(name, children)
            elif name == 'Expr':
                return create_node(ast_node.value)
            elif ast_node.__class__.__name__ in LITERALS:
                return Node(name, [])
            elif name == 'Compare':
                return generate_compare_node(ast_node)
            else:
                children = generate_children([value for _,value in ast.iter_fields(ast_node)])
                name = 'method' if name == 'Call' else name
                return Node(name, children)  
        else:
            return Node(str(ast_node), [])
        
    def generate_compare_node(ast_node):
        if len(ast_node.ops) == 1:
            name = ast_node.ops[0].__class__.__name__
            children = generate_children([ast_node.left, ast_node.comparators[0]])
            return Node(name, children)
        else:
            first_left = Node(ast_node.ops[0].__class__.__name__, [create_node(ast_node.left), create_node(ast_node.comparators[0])])
            root = Node('And', [first_left])
            current_node = root
            ops = []
            for i in range(len(ast_node.ops)-1):
                ops.append((ast_node.ops[i+1], ast_node.comparators[i], ast_node.comparators[i+1]))
            while len(ops) > 1:
                next_op = ops.pop()
                next_left = Node(next_op[0].__class__.__name__, [create_node(op) for op in [next_op[1], next_op[2]]])
                next_and = Node('And', [next_left])
                current_node.add_child(next_and)
                current_node = next_and
            right_node = Node(ops[0][0].__class__.__name__, [create_node(op) for op in [ops[0][1], ops[0][2]]])
            current_node.add_child(right_node)
            return root
    
    ast_tree = ast.parse(code_string)
    root = create_node(ast_tree)
    return root

def clear_visited(node):
    node.visited = False
    for child in node.children:
        clear_visited(child)

def get_path(terminal, pair_terminal, root, max_path_length=9):
    def helper(node, dest_node, path='', path_length=0):
        node.visited = True
        if node == dest_node:
            return path
        if path_length > max_path_length:
            return False
        path += node.name + '|'
        for child in node.children:
            if not child.visited:
                output_path = helper(child, dest_node, path=path, path_length=path_length+1)
                if output_path:
                    return output_path
        if node.parent and not node.parent.visited:
            output_path = helper(node.parent, dest_node, path=path, path_length=path_length+1)
            if output_path:
                return output_path
        return False

    clear_visited(root)
    terminal.visited = True
    path = helper(terminal.parent, pair_terminal)
    clear_visited(root)
    return path

def convert_code_to_ast_paths(code_string, max_path_length=9):
    ast_tree = create_ast_tree(code_string)
    terminals = ast_tree.get_terminals()
    paths = []
    for i in range(len(terminals)-1):
        terminal = terminals[i]
        for j in range(i+1,len(terminals)):
            pair_terminal = terminals[j]
            path = get_path(terminal, pair_terminal, ast_tree, max_path_length=max_path_length)
            if path:
                path = ','.join((terminal.name, path[:-1], pair_terminal.name))
                paths.append(path)
    return paths

def get_file(filename, latin=False):
    if latin:  
        file = open(filename, 'r', encoding='latin-1')
        return file.read()
    else:
        file = open(filename, 'r')
        return file.read()
    
def form_code_sample(sample):
    sample = sample.split('DCNL')
    sample = '\n'.join([line.strip().replace('DCSP', ' ').replace('DCTB', '\t') for line in sample])
    return sample

def append_to_file(filename, line):
    file = open(filename, 'a+')
    file.write(line + '\n')
    file.close()