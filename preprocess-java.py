import javalang
import pptree
from os import listdir
from os.path import isfile, isdir, join
import re

# Extract all methods in a .java file
# Input: string representation of a java file
# Output: array of extracted methods in the format
#         { name: <method name>,
#           search_key: <regex search query for method declaration> }
def extract_methods(code):
    tree = javalang.parse.parse(code)
    def find_method(node, methods=None):
        if not methods:
            methods = []
        if isinstance(node, list):
            for item in node:
                methods = find_method(item, methods=methods)
        else:
            if get_name(node) == 'MethodDeclaration' and node.body:
                methods.append({ 'name': node.name, 'search_key': get_method_regex_search_query(node) })
            if hasattr(node, 'attrs'):
                for attr in node.attrs:
                    methods = find_method(getattr(node, attr), methods=methods)
        return methods
    return find_method(tree)

# Given a method name, extract the body from a .java file
# Input:
#    code: a list of strings representing lines of code in a java file
#    method: method in the format returned by extract_methods
# Output:
#    string containing the method body from source
#    If the method is not found, returns False
def get_method_body(code, method):
    # code must be a list of lines
    start = -1
    end = -1
    delimeter_count = 0
    delimeter_start = False
    for idx,line in enumerate(code):
        if re.search(method['search_key'], line):
            start = idx
        if start >= 0:
            delimeter_count += line.count('{')
            if delimeter_count > 0:
                delimeter_start = True
            delimeter_count -= line.count('}')
        if delimeter_start and delimeter_count == 0:
            end = idx
            break
    if start == -1 or end == -1:
        return False
    return method['name'],'\n'.join(code[start:end+1])

# Returns all the important attributes of a javalang tree node
# for AST parsing
# Input:
#    node: javalang tree node
# Output:
#    list of attributes that should be looked at when creating a
#    code-summarization AST Node
def get_usable_attrs(node):
    name = node.__class__.__name__
    mapping = {
        'CompilationUnit': ['types'],
        'ClassDeclaration': ['name', 'body'],
        'MethodDeclaration': ['name', 'return_type', 'parameters', 'body'],
        'ReferenceType': ['name'],
        'FormalParameter': ['type', 'name'],
        'AssertStatement': ['condition'],
        'BinaryOperation': ['operator', 'operatandl', 'operatandr'],
        'MemberReference': ['member', 'qualifier'],
        'Literal': ['value'],
        'IfStatement': ['condition', 'then_statment', 'else_statement'],
        'ReturnStatement': ['expression'],
        'LocalVariableDeclaration': ['type', 'declarators'],
        'VariableDeclarator': ['name', 'initializer'],
        'MethodInvocation': ['arguments', 'member', 'qualifier'],
        'StatementExpression': ['expression']
    }
    return mapping[name]

# Creates a RegEx search query for a given method node
# Input:
#    node: a javalang tree node of type MethodDeclaration
# Output:
#    a regex string that can be used to find a method in source code
def get_method_regex_search_query(node):
    return_type = 'void' if get_name(node.return_type) == 'NoneType' else node.return_type.name 
    key = r'' + return_type + r'(<.+>)? ' + node.name
    return key

