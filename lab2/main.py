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
            self.print('Input BNF Expression')
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


def DPLL(sentences, verbose=False):
    # Helper
    def print_clauses(clauses):
        if len(clauses) == 0:
            print('Clauses: EMPTY\n')
            return
        else:
            print()
            for clause in clauses:
                if clause: print(' '.join(clause))
                else: print('Empty Clause')
        print()

    def print_literals(literals):
        for key in sorted(literals.keys()):
            print(f'{key} = {literals[key]}')

    # Eval
    def evaluate_literal(literals, literal):
        if is_negated(literal):
            return not literals[literal[1:]]
        else:
            return literals[literal]

    def evaluate_clause(literals, clause):
        return any(evaluate_literal(literals, literal) for literal in clause)

    # Functional getters
    def get_pure_literal(clauses, literals):
        all_literals = set(sum(clauses, []))
        for literal in sorted(literals.keys()):
            positive_occurance = literal in all_literals
            negative_occurance = negate(literal) in all_literals
            if positive_occurance ^ negative_occurance:  # XOR
                if positive_occurance:
                    return (negate(literal),
                            False) if is_negated(literal) else (literal, True)
                else:
                    return (negate(literal),
                            True) if is_negated(literal) else (literal, False)
        return None, None

    def get_unit_clause(clauses):
        for clause in clauses:
            if len(clause) == 1:
                return key_of(clause[0]), not is_negated(clause[0])
        return None, None

    def get_unassigned_literal(literals):
        for key in sorted(literals.keys()):
            if literals[key] == None:
                return key
        return None

    def propogate(prev_clauses, target, value, reason=None):
        to_be_removed = []
        clauses = deepcopy(prev_clauses)

        def mark_true_clause_for_removal(c, l):
            if verbose:
                print(
                    f'{reason}: remove True clause ({" ".join(c)}) containing literal {l}'
                )
            to_be_removed.append(c)

        def remove_false_literal(c, l):
            if verbose:
                print(
                    f'{reason}: remove False literal {l} from clause ({" ".join(c)})'
                )
            c.remove(l)

        if verbose:
            print(f'Propogate with {target} = {str(value)} due to {reason}')

        for c in clauses:
            if target in c:
                if value:
                    mark_true_clause_for_removal(c, target)
                    continue
                else:
                    remove_false_literal(c, target)

            negation = negate(target)
            if negation in c:
                if value:
                    remove_false_literal(c, negation)
                else:
                    mark_true_clause_for_removal(c, negation)
                    continue

        for c in to_be_removed:
            clauses.remove(c)

        if verbose:
            print_clauses(clauses)

        return clauses

    def recur(clauses, literals, verbose=False, depth=0):
        if verbose:
            print(f'\n======== DPLL Depth {depth} ========')

        if not get_unassigned_literal(literals) and all(
                evaluate_clause(literals, c) for c in clauses):
            if verbose: print('\n\nSATISFIABLE')
            print_literals(literals)
            return True

        elif not get_unassigned_literal(literals) and not any(
                evaluate_clause(literals, c) for c in clauses):
            if verbose: print('\nUNSATISFIABLE')
            return False

        if len(clauses) == 0:
            for literal in literals.keys():
                if literals[literal] == None:
                    literals[literal] = False
                    if verbose:
                        print(f'Assign unbounded {literal} = False')
            if verbose: print('\n\nSATISFIABLE')
            print_literals(literals)
            return True

        p, v = get_pure_literal(clauses, literals)
        if p:
            new_clauses = propogate(clauses, p, v, 'Pure literal elimination')
            new_literals = literals.copy()
            new_literals[p] = v
            return recur(new_clauses, new_literals, verbose, depth + 1)

        p, v = get_unit_clause(clauses)
        if p:
            new_clauses = propogate(clauses, p, v, 'Unit propagation')
            new_literals = literals.copy()
            new_literals[p] = v
            return recur(new_clauses, new_literals, verbose, depth + 1)

        p = get_unassigned_literal(literals)
        if p:
            new_literals = literals.copy()
            new_literals[p] = True
            if recur(propogate(clauses, p, True, 'branching True'),
                     new_literals, verbose, depth + 1):
                return True
            else:
                new_literals[p] = False
                return recur(propogate(clauses, p, False, 'branching False'),
                             new_literals, verbose, depth + 1)

    def init(sentences):
        clauses = []
        literals = dict()
        if verbose:
            print('DPLL CNF Input: ')
        for sentence in sentences:
            clause = sentence.strip().split(' ')

            for literal in clause:
                if negate(literal) in clause:
                    if verbose:
                        print('Skip clause with contradiction: ', sentence)
                    break
            else:
                for literal in clause:
                    literals[key_of(literal)] = None

                clauses.append(clause)
        if verbose:
            print_clauses(clauses)
        return clauses, literals

    clauses, literals = init(sentences)

    return recur(clauses, literals, verbose)


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'cnf':
        bnf_list = parse_file(args.input_file, bnf_parser)
        cnf_list = sum([bnf.to_cnf(verbose=args.v) for bnf in bnf_list], [])
        for clause in cnf_list:
            print(clause.to_cnf_str())

    elif args.mode == 'dpll':
        if not DPLL(parse_file(args.input_file, lambda x: x), verbose=args.v):
            print('UNSATISFIABLE')

    elif args.mode == 'solver':
        bnf_list = parse_file(args.input_file, bnf_parser)
        cnf_list = sum([bnf.to_cnf(verbose=args.v) for bnf in bnf_list], [])
        if not DPLL([s.to_cnf_str() for s in cnf_list], verbose=args.v):
            print('UNSATISFIABLE')

    elif args.mode == 'test':
        pass
