import argparse
import heapq
import queue as libqueue
from copy import deepcopy

# BNF Tree
BNF_OPERATORS = ['<=>', '=>', '|', '&', '!']


class BNFTreeNode:
    def __init__(self, left=None, val=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        return self.val

    # Visualize the expression tree
    def visualize(self, print_output=True):
        def _build_tree_string(root, curr_index, index=False, delimiter='-'):
            if root is None:
                return [], 0, 0, 0

            line1 = []
            line2 = []
            if index:
                node_repr = '{}{}{}'.format(curr_index, delimiter, root.val)
            else:
                node_repr = str(root.val)

            new_root_width = gap_size = len(node_repr)

            # Get the left and right sub-boxes, their widths, and root repr positions
            l_box, l_box_width, l_root_start, l_root_end = \
                _build_tree_string(root.left, 2 * curr_index + 1, index, delimiter)
            r_box, r_box_width, r_root_start, r_root_end = \
                _build_tree_string(root.right, 2 * curr_index + 2, index, delimiter)

            # Draw the branch connecting the current root node to the left sub-box
            # Pad the line with whitespaces where necessary
            if l_box_width > 0:
                l_root = (l_root_start + l_root_end) // 2 + 1
                line1.append(' ' * (l_root + 1))
                line1.append('_' * (l_box_width - l_root))
                line2.append(' ' * l_root + '/')
                line2.append(' ' * (l_box_width - l_root))
                new_root_start = l_box_width + 1
                gap_size += 1
            else:
                new_root_start = 0

            # Draw the representation of the current root node
            line1.append(node_repr)
            line2.append(' ' * new_root_width)

            # Draw the branch connecting the current root node to the right sub-box
            # Pad the line with whitespaces where necessary
            if r_box_width > 0:
                r_root = (r_root_start + r_root_end) // 2
                line1.append('_' * r_root)
                line1.append(' ' * (r_box_width - r_root + 1))
                line2.append(' ' * r_root + '\\')
                line2.append(' ' * (r_box_width - r_root))
                gap_size += 1
            new_root_end = new_root_start + new_root_width - 1

            # Combine the left and right sub-boxes with the branches drawn above
            gap = ' ' * gap_size
            new_box = [''.join(line1), ''.join(line2)]
            for i in range(max(len(l_box), len(r_box))):
                l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
                r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
                new_box.append(l_line + gap + r_line)

            # Return the new box, its width and its root repr positions
            return new_box, len(new_box[0]), new_root_start, new_root_end

        lines = _build_tree_string(self, 0, False, '-')[0]
        output = '\n' + '\n'.join((line.rstrip() for line in lines))
        if print_output: print(output)
        return output

    def print(self, title=None, op=['<=>', '=>', '|', '&']):
        def _recur(root):
            if not root: return
            if root.left:
                print('(', end='')
                _recur(root.left)
            print(f' {root.val} ' if root.val in op else root.val, end='')
            if root.right:
                _recur(root.right)
                if not root.val == '!': print(')', end='')

        if title: print(title, end=':\t')
        _recur(self)
        print()

    def to_cnf_str(self, title=None, op=['|']):
        def _recur(root):
            if not root: return ''
            return _recur(root.left) + (' ' if root.val in op else
                                        root.val) + _recur(root.right)

        return _recur(self)

    def preorder_call(self, func, op=BNF_OPERATORS):
        def _recur(node: BNFTreeNode):
            nonlocal func
            if node and node.val in op:
                node = func(node)
                node.left = _recur(node.left)
                node.right = _recur(node.right)
            return node

        return _recur(self)

    def satisfy_condition(self, condition):
        q = libqueue.Queue()
        q.put(self)
        while not q.empty():
            node = q.get()
            if condition(node):
                return True
            else:
                if node.left: q.put(node.left)
                if node.right: q.put(node.right)

    def currify_preorder_call(self, *func_list, verbose=False):
        res = self
        for func in func_list:
            res = res.preorder_call(func)
            if verbose: res.print(func.__name__)
            if verbose: res.visualize()
        return res

    def to_cnf(self, verbose=False):
        def eliminate_iff(node: BNFTreeNode):
            if not node.val == '<=>': return node
            return BNFTreeNode(BNFTreeNode(node.left, '=>', node.right), '&',
                               BNFTreeNode(node.right, '=>', node.left))

        def eliminate_implication(node: BNFTreeNode):
            if not node.val == '=>': return node
            return BNFTreeNode(BNFTreeNode(None, '!', node.left), '|',
                               node.right)

        def apply_demorgans_law(node: BNFTreeNode):
            if node.val == '!':
                if node.right.val == '|':
                    return BNFTreeNode(
                        BNFTreeNode(None, '!', node.right.left), '&',
                        BNFTreeNode(None, '!', node.right.right))
                if node.right.val == '&':
                    return BNFTreeNode(
                        BNFTreeNode(None, '!', node.right.left), '|',
                        BNFTreeNode(None, '!', node.right.right))
                if node.right.val == '!':
                    return node.right.right
            return node

        def distribute_or(node: BNFTreeNode):
            if node.val == '|' and node.right.val == '&':
                return BNFTreeNode(
                    BNFTreeNode(node.left, '|', node.right.left), '&',
                    BNFTreeNode(node.left, '|', node.right.right))
            if node.val == '|' and node.left.val == '&':
                return BNFTreeNode(
                    BNFTreeNode(node.left.left, '|', node.right), '&',
                    BNFTreeNode(node.left.right, '|', node.right))
            return node

        def seperate_conjunctions(root: BNFTreeNode):
            unprocessed = libqueue.Queue()
            unprocessed.put(root)
            res = []
            while not unprocessed.empty():
                node = unprocessed.get()

                if node.val == '&':
                    if node.left: unprocessed.put(node.left)
                    if node.right: unprocessed.put(node.right)
                else: res.append(node)
            return res

        def is_contradiction(root: BNFTreeNode):
            positive = set()
            negative = set()

            def _recur(node: BNFTreeNode):
                if not node: return False

                if node.val == '|':
                    return _recur(node.left) or _recur(node.right)

                elif node.val == '!':
                    if node.right.val in positive:
                        return True
                    negative.add(node.right.val)

                else:  # leaf
                    if node.val in negative:
                        return True
                    positive.add(node.val)

                return False

            res = _recur(root)
            if res and verbose: root.print('remove contradiction')
            return res

        if verbose:
            self.print('Input expression')
            self.visualize()

        res = self.currify_preorder_call(eliminate_iff,
                                         eliminate_implication,
                                         apply_demorgans_law,
                                         verbose=verbose)

        while res.satisfy_condition(lambda node: node.val == '|' and (
                node.right.val == '&' or node.left.val == '&')):
            res = res.currify_preorder_call(distribute_or, verbose=verbose)

        cnf_list = seperate_conjunctions(res)

        cnf_list = [s for s in cnf_list if not is_contradiction(s)]

        return cnf_list


# Argument & Input File Parser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', help="verbose mode")
    parser.add_argument("-t", action='store_true', help="test mode")
    parser.add_argument("-mode",
                        help="Program's mode",
                        choices=['cnf', 'dpll', 'solver', 'test'])
    parser.add_argument(
        'input_file',
        help='A mode-dependent input file',
        default='ex1.txt',
    )
    return parser.parse_args()


def bnf_parser(s: str, op=BNF_OPERATORS) -> BNFTreeNode:
    if not op:
        return BNFTreeNode(val=s.strip())
    idx = s.rfind(op[0])  # rightmost
    if idx == -1:
        # 1st operator not found
        return bnf_parser(s, op[1:])
    return BNFTreeNode(bnf_parser(s[:idx]), op[0],
                       bnf_parser(s[idx + len(op[0]):]))


def parse_file(filename, parser):
    with open(filename) as f:
        return [parser(line.strip()) for line in f.readlines()]


# CNF Helper Functions


def is_unit_clause(clause: list) -> bool:
    return len(clause) == 1


def is_negated(literal: str) -> bool:
    return literal[0] == '!'


def negate(literal: str) -> str:
    if is_negated(literal):
        return literal[1:]
    return '!' + literal


def key_of(literal: str) -> str:
    return literal.replace('!', '')


def print_list(l, title, sep=','):
    if not title: print(sep.join(l))
    else: print(f'{title}: {sep.join(l)}')


# CNF Data Structure
class CNF:
    literals = dict()
    clauses = []
    unassigned_literals = set()

    def __init__(self, sentences, verbose=False) -> None:
        self.verbose = verbose
        if verbose: print('Parsing CNF input')
        for sentence in sentences:
            clause = sentence.strip().split(' ')

            for literal in clause:
                if negate(literal) in clause:
                    if verbose:
                        print('Skip clause with contradiction: ', sentence)
                    break
            else:
                for literal in clause:
                    self.assign_literal(literal, None)

                self.clauses.append(clause)

    def __str__(self):
        return '\n'.join([
            'Literals: ' + str(self.literals), 'Clauses: ', '\n'.join(
                [f'{" ".join(clause)}' for clause in self.clauses])
        ]) + '\n'

    def print_clauses(self):
        if len(self.clauses) == 0:
            print('Clauses: EMPTY\n')
            return
        else:
            print('Clauses:')
            for clause in self.clauses:
                print(' '.join(clause))
        print()

    def print_literals(self, reason=None):
        if reason: print(f'{reason}:', end=': ')
        for key, value in self.literals.items():
            print(f'{key} = {value}')

    def assign_literal(self, literal, value):
        key = literal.replace('!', '')

        if value is None:
            self.unassigned_literals.add(key)

        elif key in self.unassigned_literals:
            self.unassigned_literals.remove(key)

        self.literals[key] = value

    def evaluate_literal(self, literal):
        if is_negated(literal):
            return not self.literals[literal[1:]]
        else:
            return self.literals[literal]

    def evaluate_clause(self, clause):
        return any(self.evaluate_literal(literal) for literal in clause)

    def evaluate(self):
        return all(self.evaluate_clause(clause) for clause in self.clauses)

    def get_pure_literals(self) -> list:
        all_literals = set(sum(self.clauses, []))
        pure_literals = set()
        for literal in self.literals.keys():
            positive_occurance = literal in all_literals
            negative_occurance = negate(literal) in all_literals
            if positive_occurance ^ negative_occurance:  # XOR
                if positive_occurance:
                    pure_literals.add(literal)
                else:
                    pure_literals.add('!' + literal)

        return sorted(list(pure_literals))

    def propogate(self, literal, value, reason=None):
        to_be_removed = []

        def mark_true_clause_for_removal(c, literal):
            if self.verbose:
                print(
                    f'{reason}: remove True clause ({" ".join(c)}) containing literal {literal}'
                )
            to_be_removed.append(c)

        def remove_false_literal(c, l):
            if self.verbose:
                print(
                    f'{reason}: remove False literal {l} from clause ({" ".join(c)})'
                )
            c.remove(l)

        self.assign_literal(literal, value)
        if self.verbose:
            print(f'Propogate with {literal} = {str(value)} due to {reason}')

        for c in self.clauses:
            if literal in c:
                if value:
                    mark_true_clause_for_removal(c, literal)
                    continue
                else:
                    remove_false_literal(c, literal)
            negation = negate(literal)
            if negation in c:
                if value:
                    remove_false_literal(c, negation)
                else:
                    mark_true_clause_for_removal(c, negation)
                    continue

        for c in to_be_removed:
            self.clauses.remove(c)

        if self.verbose:
            self.print_clauses()

    def dpll(self, i=0) -> bool:
        if self.verbose:
            print(f'\n======== Depth {i} ========')
            print(self)

        if len(self.clauses) == 0:
            for literal in self.literals.keys():
                if self.literals[literal] == None:
                    self.assign_literal(literal, False)
                    self.print_literals('SSS')
                    if self.verbose:
                        print(f'Assign unbounded {literal} = False')
            return True
        elif any(len(clause) == 0 for clause in self.clauses):
            if self.verbose: print('Contradiction! Backtrack')
            return False

        # Easy Case: unit propagation
        for clause in self.clauses:
            if is_unit_clause(clause):
                ucl = clause[0]
                self.clauses.remove(clause)
                self.propogate(key_of(ucl), not is_negated(ucl),
                               'Unit propagation')
                if len(self.clauses) == 0:
                    for literal in self.literals.keys():
                        if self.literals[literal] == None:
                            self.assign_literal(literal, False)
                            # self.print_literals('unit propagation')
                            if self.verbose:
                                print(f'Assign unbounded {literal} = False')
                    return True
                elif any(len(clause) == 0 for clause in self.clauses):
                    if self.verbose: print('Contradiction! Backtrack')
                    return False

        # Easy Case: pure literal elimination
        for pl in self.get_pure_literals():
            self.propogate(key_of(pl), not is_negated(pl),
                           'Pure literal elimination')
            if len(self.clauses) == 0:
                for literal in self.literals.keys():
                    if self.literals[literal] == None:
                        # self.print_literals('pure literal elimination')
                        self.assign_literal(literal, False)
                        # self.print_literals('pure literal elimination')
                        if self.verbose:
                            print(f'Assign unbounded {literal} = False')
                return True
            elif any(len(clause) == 0 for clause in self.clauses):
                if self.verbose: print('Contradiction! Backtrack')
                return False

        # Hard Case
        literal = sorted(self.unassigned_literals)[0]
        self.unassigned_literals.remove(literal)

        false_branch = deepcopy(self)
        false_branch.clauses = deepcopy(self.clauses)
        false_branch.literals = deepcopy(self.literals)
        false_branch.unassigned_literals = deepcopy(self.unassigned_literals)

        if self.verbose: print('Branch at literal', literal)
        self.propogate(literal, True, 'branching True')

        if self.dpll(i + 1): return True

        else:
            false_branch.propogate(literal, False, 'branching False')
            return false_branch.dpll(i + 1)


def print_clauses(clauses):
    if len(clauses) == 0:
        print('Clauses: EMPTY\n')
        return
    else:
        print('Clauses:')
        for clause in clauses:
            print(' '.join(clause))
    print()


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'cnf':
        bnf_list = parse_file(args.input_file, bnf_parser)
        cnf_list = sum([bnf.to_cnf(verbose=args.v) for bnf in bnf_list], [])
        for clause in cnf_list:
            print(clause.to_cnf_str())

    elif args.mode == 'dpll':
        cnf = CNF(parse_file(args.input_file, lambda x: x), verbose=args.v)
        if args.v:
            cnf.print_clauses()

        sat = cnf.dpll()

        if sat:
            if args.v: print(f'Satisfiable: {sat}')
            cnf.print_literals()
        else:
            print('Unsatisfiable')

    elif args.mode == 'solver':
        bnf_list = parse_file(args.input_file, bnf_parser)
        cnf_list = sum([bnf.to_cnf(verbose=args.v) for bnf in bnf_list], [])

        cnf = CNF([s.to_cnf_str() for s in cnf_list], verbose=args.v)

        if args.v:
            cnf.print_clauses()

        sat = cnf.dpll()
        if sat:
            if args.v: print(f'Satisfiable: {sat}')
            cnf.print_literals()
        else:
            print('Unsatisfiable')
    elif args.mode == 'test':
        # bnf_list = parse_file(args.input_file, bnf_parser)
        # cnf_list = sum([bnf.to_cnf(verbose=args.v) for bnf in bnf_list], [])
        # cnf = CNF([s.to_cnf_str() for s in cnf_list], verbose=args.v)
        cnf = CNF(parse_file(args.input_file, lambda x: x), verbose=args.v)
        cnf.literals = {
            'P': True,
            'Q': True,
            'R': True,
            'W': False,
            'U': False,
            'X': True
        }
        # cnf.literals = {
        #     'P': False,
        #     'Q': False,
        #     'R': False,
        #     'W': False,
        #     'X': False
        # }
        # cnf.literals = {
        #     'A': True,
        #     'B': True,
        #     'C': False,
        #     'P': False,
        #     'Q': True,
        #     'W': False
        # }
        cnf.print_literals()
        for clause in cnf.clauses:
            print(' '.join(clause), ' => ', cnf.evaluate_clause(clause))

        print('Sat: ', cnf.evaluate())
