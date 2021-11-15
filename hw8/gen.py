GW = [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], ['I', 'J', 'K', 'L'],
      ['M', 'N', 'P', 'Q']]
idx = dict((j, (x, y)) for x, i in enumerate(GW) for y, j in enumerate(i))

states = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P',
    'Q'
]

actions = dict((s, [None] * 4) for s in states)

l = len(GW) - 1

for row_num, row in enumerate(GW):
    for col_num, item in enumerate(row):
        # up
        actions[item][0] = item if row_num == 0 else GW[row_num -
                                                        1][col_num]  # row 0
        # down
        actions[item][1] = item if row_num == l else GW[row_num +
                                                        1][col_num]  # row l
        # left
        actions[item][2] = item if col_num == 0 else GW[row_num][col_num -
                                                                 1]  # col 0
        # right
        actions[item][3] = item if col_num == l else GW[row_num][col_num +
                                                                 1]  # col l

# print(actions)

# for key, values in actions.items():
#     print(f'{key} : [{", ".join(values)}]')
#     print(f'{key} % .25 .25 .25 .25')

# print('\n'.join([f'{key} : [{s(values)}]' for key, values in actions.items()]))


def print_dict_list(d):
    print('\n'.join([f'{k} : [{", ".join(v)}]' for k, v in d.items()]))


def print_dict_3f(d):
    # print('\n'.join([f'{k} : {round(v, 3) if v is float else v}]' for k, v in d.items()]))
    print('\n'.join(['{:s} : {:.3f}'.format(k, v) for k, v in d.items()]))


def converged(d1, d2, tol=0.001):
    for k in d1.keys():
        if abs(d1[k] - d2[k]) > tol:
            return False
    return True


import numpy as np


def chance_node_markov_reward_process(states,
                                 actions,
                                 values,
                                 probs=[0.25] * 4,
                                 tol=0.001,
                                 max_iter=150):
    new_values = values.copy()
    for i in range(max_iter):
        # print(f'Iteration {i}')
        prev_values = new_values.copy()
        for s in states:
            if s == 'A' or s == 'Q':
                continue
            action_values = [prev_values[a] for a in actions[s]]
            new_values[s] = values[s] + np.dot(probs, action_values)
            # print(f'new_values[{s}] = {new_values[s]}')

    print(f'Max iterations {max_iter} reached')
    return new_values


def chance_node_value_iter(states,
                           actions,
                           values,
                           probs=[0.25] * 4,
                           tol=0.001,
                           max_iter=150):
    new_values = values.copy()
    for i in range(max_iter):
        prev_values = new_values.copy()
        for s in states:

            action_values = [prev_values[a] for a in actions[s]]
            new_values[s] = max(new_values[s], np.dot(probs, action_values))

        if converged(prev_values, new_values, tol):
            print(f'Converged in {i} iterations')
            return new_values

    print(f'Max iterations {max_iter} reached')
    return new_values


def q1():
    values = dict((i, -2) for i in states)
    values['A'] = 1
    values['Q'] = 1
    # new_values = chance_node_value_iter(states, actions, values)
    new_values = chance_node_markov_reward_process(states, actions, values)
    # print_dict_3f(values)
    # print('\n\n')
    print_dict_3f(new_values)


def decision_node_value_iter(states,
                             actions,
                             values,
                             df=0.9,
                             alpha=0.15,
                             tol=0.001,
                             max_iter=150):
    p_success = 1 - alpha

    def probs(decision):
        nonlocal actions, alpha
        n = len(actions[decision])
        p_fail = alpha / n

        return [
            p_success if decision == a else p_fail for a in actions[decision]
        ]

    new_values = values.copy()
    for i in range(max_iter):
        prev_values = new_values.copy()
        for s in states:

            action_values = [prev_values[a] for a in actions[s]]

            for decision in actions[s]:
                q = values[decision] + df * np.dot(probs(decision), action_values)

            new_values[s] = max(new_values[s], np.dot(actions[s],
                                                      action_values))

        if converged(prev_values, new_values, tol):
            print(f'Converged in {i} iterations')
            return new_values

    print(f'Max iterations {max_iter} reached')
    return new_values


def q2():
    values = dict((i, -1) for i in states)
    values['A'] = 20
    values['Q'] = -20
    values['G'] = 2
    values['J'] = 2

    new_values = chance_node_value_iter(states, actions, values)
    # print_dict_3f(values)
    # print('\n\n')
    print_dict_3f(new_values)


if __name__ == '__main__':
    q1()
    # q2()


# Value iteration