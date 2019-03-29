import ast
import pptree
import sys

class Node:
    def __init__(self, name, children, parent=None):
        self.name = name
        self.children = []
        for child in children:
            if child is not None:
                self.children.append(child)
        self.parent = parent
        self.visited = False
        for child in self.children:
            child.parent = self

    def add_child(self, child):
        self.children.append(child)

def generate_nodes_from_fields(fields_iter):
    nodes = []
    for field,value in fields_iter:
        if field != 'decorator_list' and field != 'returns' and field != 'annotation' and field != 'ctxex':
            nodes.append(create_node(value))
    return nodes
            

def create_node(ast_node):
    if isinstance(ast_node, ast.AST):
        name = ast_node.__class__.__name__
        if name == 'BinOp':
            name = ast_node.op.__class__.__name__
            children = [create_node(ast_node.left), create_node(ast_node.right)]
            return Node(name, children)
        elif name == 'Name':
            return create_node(ast_node.id)
        elif name == 'Expr':
            name = ast_node.value.__class__.__name__
            children = generate_nodes_from_fields(ast.iter_fields(ast_node.value))
            return Node(name, children) 
        elif name == 'BoolOp':
            name = ast_node.op.__class__.__name__
            children = [create_node(child) for child in ast_node.values]
            return Node(name, children)
        else:
            children = generate_nodes_from_fields(ast.iter_fields(ast_node))
            return Node(name, children)  
    elif isinstance(ast_node, list):
        if len(ast_node) == 0:
            return None
        if len(ast_node) == 1:
            return create_node(ast_node[0])
        name = 'block'
        children = [create_node(node) for node in ast_node]
        return Node(name, children)
    else:
        return Node(str(ast_node), [])

def create_ast_tree(code_string):
    ast_tree = ast.parse(code_string)
    root = create_node(ast_tree)
    return root

def print_ast_tree(code):
    root = create_ast_tree(code)
    pptree.print_tree(root, childattr='children', nameattr='name')

def find_terminals(node, terminals=None):
    if terminals == None:
        terminals = []
    if len(node.children) == 0:
        terminals.append(node)
    else:
        for child in node.children:
            find_terminals(child, terminals=terminals)
    return terminals

def clear_visited(node):
    node.visited = False
    for child in node.children:
        clear_visited(child)

def get_path_helper(node, dest_node, path=''):
    node.visited = True
    if node == dest_node:
        return path
    path += node.name + '|'
    for child in node.children:
        if not child.visited:
            output_path = get_path_helper(child, dest_node, path=path)
            if output_path:
                return output_path
    if not node.parent.visited:
        output_path = get_path_helper(node.parent, dest_node, path=path)
        if output_path:
            return output_path
    return False

def get_path(terminal, pair_terminal, root):
    clear_visited(root)
    terminal.visited = True
    path = get_path_helper(terminal.parent, pair_terminal)
    clear_visited(root)
    return path

def convert_code_to_ast_paths(code_string):
    ast_tree = create_ast_tree(code_string)
    terminals = find_terminals(ast_tree)
    paths = []
    for i in range(len(terminals)-1):
        terminal = terminals[i]
        for j in range(i+1,len(terminals)):
            pair_terminal = terminals[j]
            path = get_path(terminal, pair_terminal, ast_tree)
            path = '|'.join((terminal.name, path[:-1], pair_terminal.name))
            paths.append(path)
    return paths

if __name__ == '__main__':
    filename = sys.argv[1]
    txt = open(filename, 'r')
    source = txt.read()
    ast_paths = convert_code_to_ast_paths(source)
