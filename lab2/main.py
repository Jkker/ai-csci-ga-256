import argparse
import sys
from math import sqrt
import queue as lib_queue
from types import TracebackType


class N:
    def __init__(self, left=None, val=None, right=None):
        self.val = val  # Assign data
        self.left = left  # Initialize
        self.right = right  # Initialize

    # Visualize the expression tree
    def __str__(self):
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


def inorder(root: N, op=['<=>', '=>', '|', '&']):
    if root.left:
        print('(', end='')
        inorder(root.left)
    print(f' {root.val} ' if root.val in op else root.val, end='')
    if root.right:
        inorder(root.right)
        print(')', end='')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', help="verbose mode")
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


def bnf2cnf(bnf_expr):
    def iff(node: N):
        if node.val == '<=>':
            return N(N(node.left, '->', node.right), '&',
                     N(node.right, '->', node.left))
        return node

    def implication(node: N):
        if node.val == '=>':
            return N(N(None, '!', node.left), '|', node.right)
        return node

    pass


if __name__ == '__main__':
    args = get_args()
    expr_list = parse_expr_file(args.input_file)

    idx = 0
    print(expr_list[idx])
    # inorder(expr_list[idx])
    inorder(expr_list[idx])

    if args.mode == 'cnf':
        pass
    elif args.mode == 'dpll':
        pass
    elif args.mode == 'solver':
        pass