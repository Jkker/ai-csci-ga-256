import argparse
import sys
from math import sqrt
import queue as lib_queue
from types import TracebackType
from copy import deepcopy


class N:
    def __init__(self, left=None, val=None, right=None):
        self.val = val  # Assign data
        self.left = left  # Initialize
        self.right = right  # Initialize

    def __str__(self):
        return self.val

    # Visualize the expression tree
    def print_tree(self):
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
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    def print_inorder(self, op=['<=>', '=>', '|', '&']):
        def _recur(root):
            if not root: return
            if root.left:
                print('(', end='')
                _recur(root.left)
            print(f' {root.val} ' if root.val in op else root.val, end='')
            if root.right:
                _recur(root.right)
                print(')', end='')

        _recur(self)
        print()

    def toCNF(self):
        def iff(node: N):
            if not node.val == '<=>': return node

            return N(N(node.left, '=>', node.right), '&',
                     N(node.right, '=>', node.left))

        def implication(node: N):
            if not node.val == '=>': return node
            return N(N(None, '!', node.left), '|', node.right)

        def de_morgan(node: N):
            if not node.val == '!': return node
            if node.right.val == '|':
                return N(N(None, '!', node.right.left), '&',
                         N(None, '!', node.right.right))
            if node.right.val == '&':
                return N(N(None, '!', node.right.left), '|',
                         N(None, '!', node.right.right))

        def _dfs(node: N, func):
            if node.val in ['<=>', '=>', '|', '&', '!']:
                node.left = _dfs(node.left, func)
                node.right = _dfs(node.right, func)
                node = func(node)
            return node

        tmp = _dfs(self, iff)
        print('Eliminated <=>')
        tmp.print_inorder()
        tmp = _dfs(tmp, implication)
        print('Eliminated =>')
        tmp.print_inorder()
        # tmp = _dfs(tmp, de_morgan)
        # tmp.print_inorder()
        return tmp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', help="verbose mode")
    parser.add_argument("-t", action='store_true', help="test mode")
    parser.add_argument("-mode",
                        help="Program's mode",
                        choices=['cnf', 'dpll', 'solver'])
    parser.add_argument(
        'input_file',
        help='A mode-dependent input file',
        default='ex1.txt',
    )
    return parser.parse_args()


def parse_expr(s: str, op=['<=>', '=>', '|', '&', '!']) -> N:
    if not op:
        return N(val=s.strip())
    idx = s.rfind(op[0])  # rightmost
    if idx == -1:
        # 1st operator not found
        return parse_expr(s, op[1:])
    return N(parse_expr(s[:idx]), op[0], parse_expr(s[idx + len(op[0]):]))


def parse_expr_file(filename):
    with open(filename) as f:
        return [parse_expr(line.strip()) for line in f.readlines()]


def test():
    # d = N(None, '!', N(N(val='A'), '&', N(val='B')))
    d = N(N(val='A'), '<=>', N(val='B'))
    d.print_inorder()
    d.toCNF().print_inorder()


if __name__ == '__main__':
    args = get_args()
    expr_list = parse_expr_file(args.input_file)
    if args.t:
        test()
    else:
        if args.mode == 'cnf':
            pass
        elif args.mode == 'dpll':
            pass
        elif args.mode == 'solver':
            pass