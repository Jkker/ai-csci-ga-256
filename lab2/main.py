import argparse


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

    def inorder_print(self, title=None, op=['<=>', '=>', '|', '&']):
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

    def preorder_func_call(self, func, op=['<=>', '=>', '|', '&', '!']):
        def _recur(node: N):
            nonlocal func
            if node and node.val in op:
                node = func(node)
                node.left = _recur(node.left)
                node.right = _recur(node.right)
            return node

        return _recur(self)

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

        def distribute_or_bfs(root: N):
            unprocessed = [root]
            res = []
            while unprocessed:
                node = unprocessed.pop()
                if node.val == '|':
                    if node.right.val == '&':
                        unprocessed.append(N(node.left, '|', node.right.left))
                        unprocessed.append(N(node.left, '|', node.right.right))
                    if node.left.val == '&':
                        unprocessed.append(N(node.left.left, '|', node.right))
                        unprocessed.append(N(node.left.right, '|', node.right))
                else:
                    res.append(node)
            return res

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
            if res and verbose: root.inorder_print('remove contradiction')
            return res

        def currify(node: N, *func_list):
            res = node
            for func in func_list:
                res = res.preorder_func_call(func)
                if verbose: res.inorder_print(func.__name__)
                if verbose: res.visualize()
            return res

        res = currify(self, eliminate_iff, eliminate_implication,
                      apply_demorgans_law, distribute_or, distribute_or)

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


def parse_bnf(s: str, op=['<=>', '=>', '|', '&', '!']) -> N:
    if not op:
        return N(val=s.strip())
    idx = s.rfind(op[0])  # rightmost
    if idx == -1:
        # 1st operator not found
        return parse_bnf(s, op[1:])
    return N(parse_bnf(s[:idx]), op[0], parse_bnf(s[idx + len(op[0]):]))


def parse_expr_file(filename):
    with open(filename) as f:
        return [parse_bnf(line.strip()) for line in f.readlines()]


def test():
    d = N(None, '!', N(N(val='A'), '&', N(val='B')))
    # d = N(N(val='A'), '<=>', N(val='B'))
    d.inorder_print()
    cnf = d.to_cnf()
    print('CNF:\t', end='')
    cnf.inorder_print()
    print()


if __name__ == '__main__':
    args = get_args()
    expr_list = parse_expr_file(args.input_file)

    if args.t:
        test()

    else:
        if args.mode == 'cnf':
            cnf_list = []
            for expr in expr_list:
                if args.v:
                    expr.inorder_print('parsed input expression')
                    expr.visualize()
                cnf_list = [*cnf_list, *expr.to_cnf(verbose=args.v)]
            for s in cnf_list:
                print(s.to_cnf_str())

        elif args.mode == 'dpll':
            pass
        elif args.mode == 'solver':
            pass