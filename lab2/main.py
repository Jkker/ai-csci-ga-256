import argparse
import queue as libqueue

BNF_OPERATORS = ['<=>', '=>', '|', '&', '!']


class N:
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
        def _recur(node: N):
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
        def eliminate_iff(node: N):
            if not node.val == '<=>': return node
            return N(N(node.left, '=>', node.right), '&',
                     N(node.right, '=>', node.left))

        def eliminate_implication(node: N):
            if not node.val == '=>': return node
            return N(N(None, '!', node.left), '|', node.right)

        def apply_demorgans_law(node: N):
            if node.val == '!':
                if node.right.val == '|':
                    return N(N(None, '!', node.right.left), '&',
                             N(None, '!', node.right.right))
                if node.right.val == '&':
                    return N(N(None, '!', node.right.left), '|',
                             N(None, '!', node.right.right))
                if node.right.val == '!':
                    return node.right.right
            return node

        def distribute_or(node: N):
            if node.val == '|' and node.right.val == '&':
                return N(N(node.left, '|', node.right.left), '&',
                         N(node.left, '|', node.right.right))
            if node.val == '|' and node.left.val == '&':
                return N(N(node.left.left, '|', node.right), '&',
                         N(node.left.right, '|', node.right))
            return node

        def seperate_conjunctions(root: N):
            unprocessed = [root]
            res = []
            while unprocessed:
                node = unprocessed.pop()

                if node.val == '&':
                    if node.left: unprocessed.append(node.left)
                    if node.right: unprocessed.append(node.right)
                else: res.append(node)
            return res

        def is_contradiction(root: N):
            positive = set()
            negative = set()

            def _recur(node: N):
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
            self.print('parsed input expression')
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


def bnf_parser(s: str, op=BNF_OPERATORS) -> N:
    if not op:
        return N(val=s.strip())
    idx = s.rfind(op[0])  # rightmost
    if idx == -1:
        # 1st operator not found
        return bnf_parser(s, op[1:])
    return N(bnf_parser(s[:idx]), op[0], bnf_parser(s[idx + len(op[0]):]))


def parse_file(filename, parser):
    with open(filename) as f:
        return [parser(line.strip()) for line in f.readlines()]


def bnf2cnf(bnf_list, verbose=False) -> list[N]:
    return sum([bnf.to_cnf(verbose=verbose) for bnf in bnf_list], [])


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'cnf':
        cnf_list = bnf2cnf(parse_file(args.input_file, bnf_parser),
                           verbose=args.v)
        for s in cnf_list:
            print(s.to_cnf_str())

    elif args.mode == 'dpll':
        pass
    elif args.mode == 'solver':
        cnf_list = bnf2cnf(parse_file(args.input_file, bnf_parser),
                           verbose=args.v)
