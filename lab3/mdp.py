import argparse
import heapq
import queue as libqueue
from copy import deepcopy
import numpy as np
# Argument & Input File Parser
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', type=float, default=1.0)
    parser.add_argument('-min', action='store_true', default=False)
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-tol', type=float, default=0.01)
    parser.add_argument('-iter', type=int, default=100)
    parser.add_argument(
        'input_file',
        help='A mode-dependent input file',
        default='ex1.txt',
    )
    args = parser.parse_args()
    if args.df < 0 or args.df > 1:
        raise ValueError('Discount factor df must be between 0 and 1')
    return args


def parse_input(input_file):
    values = {}
    edges = {}
    probs = {}
    nodes = set()
    node_type = {}

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line or line.startswith('#'):  # blank or comment
                continue

            if '=' in line:  # teriminal node
                name, value = line.split('=')
                name = name.strip()
                value = float(value.strip())
                node_type[name] = 'terminal'
                nodes.add(name)
                values[name] = value

            if '%' in line:  # decision / chance node
                name, value = line.split('%')
                name = name.strip()
                _probs = [float(p) for p in value.strip().split(' ')]
                nodes.add(name)
                node_type[name] = 'decision'
                probs[name] = _probs

            if ':' in line:  # edge
                name, value = line.split(':')
                name = name.strip()
                nodes.add(name)
                edges[name.strip()] = [
                    e.strip() for e in value.strip()[1:-1].split(',')
                ]

    return nodes, edges, values, probs


class G:
    def __init__(self,
                 states,
                 edges,
                 rewards,
                 probs,
                 df=1.0,
                 tol=0.01,
                 max_iter=100,
                 verbose=False):
        self.verbose = verbose
        self.df = df
        self.tol = tol
        self.max_iter = max_iter
        self.states = states

        # self.adj = dict(
        #     (name, dict((adj, 0) for adj in states)) for name in states)

        self.transitions = dict()
        self.rewards = rewards or {s: 0 for s in states}
        self.probs = dict()
        self.typeof = dict()
        if verbose: print('\n==== MDP INITIALIZATION ====')
        for s in states:
            if not s in probs and not s in edges and s in rewards:  # terminal node
                self.typeof[s] = 'terminal'
                if verbose: print(f'{s} (Terminal) = {rewards[s]}')
                continue

            actions = edges[s]
            self.rewards[s] = rewards[s] if s in rewards else 0

            if not s in probs:  # decision node w/o probs
                self.typeof[s] = 'decision'
                p_succ = 1
                self.transitions[s] = dict(
                    (action, [(p_succ, action)]) for action in actions)
                if verbose:
                    print(
                        f'{s} (Decision P={p_succ}) = {self.rewards[s]} : {", ".join(actions)}'
                    )
                continue

            if len(actions) == 1:  # decision node w/ only 1 neighbor
                self.typeof[s] = 'chance'
                p_succ = 1
                # self.transitions[s] = {actions[0]: 1}
                self.probs[s] = [(p_succ, actions[0])]
                if verbose:
                    print(
                        f'{s} (Decision P={p_succ}) = {self.rewards[s]} : {", ".join(actions)}'
                    )
                continue

            if len(probs[s]) == 1:  # decision node
                self.typeof[s] = 'decision'
                p_succ = probs[s][0]
                # p_fail = round((1 - p_succ) / (len(actions) - 1), 3)
                p_fail = (1 - p_succ) / (len(actions) - 1)
                self.transitions[s] = dict(
                    (action, [(p_succ if neighbor == action else p_fail,
                               neighbor) for neighbor in actions])
                    for action in actions)

                # self.rewards[s] = rewards[s] if s in rewards else 0

                if verbose:
                    print(
                        f'{s} (Decision P={p_succ}) = {self.rewards[s]} : {", ".join(actions)}'
                    )

            else:  # chance node
                self.typeof[s] = 'chance'
                self.probs[s] = [(p, outcome)
                                 for outcome, p in zip(edges[s], probs[s])]
                if verbose:
                    print(
                        f'{s} (Chance): {" ".join(f"{o}={p}" for (p, o) in self.probs[s])}'
                    )
                # self.probs[s] = dict((outcome, p) for outcome, p in zip(edges[s], probs[s]))

    def T(self, state, action=None):
        if self.typeof[state] == 'chance':
            return self.probs[state]
            # return [(p, s) for s, p in self.probs[state].items()]
        if self.typeof[state] == 'decision':
            return self.transitions[state][action]
        return [(0, state)]

    def A(self, state):
        return list(self.transitions[state].keys()
                    ) if state in self.transitions else []

    def R(self, state):
        return self.rewards[state] if state in self.rewards else 0

    def q_value(self, state, action, utility):
        if self.typeof[state] == 'terminal':
            return self.R(state)
        res = 0
        for p, s_prime in self.T(state, action):
            res += p * (self.R(state) + self.df * utility[s_prime])
        return res

    def value_eval(self, state, utility):
        if self.typeof[state] == 'terminal':
            return self.R(state)

        if self.typeof[state] == 'chance':
            return self.q_value(state, None, utility)

        if self.typeof[state] == 'decision':
            return max(self.q_value(state, a, utility) for a in self.A(state))

    def value_iter(self):
        if self.verbose: print('\n==== MDP VALUE ITERATION ====')

        values = {s: 0 for s in self.states}
        for i in range(self.max_iter):
            curr_values = deepcopy(values)

            delta = 0

            for s in self.states:

                values[s] = self.value_eval(s, curr_values)

                delta = max(delta, abs(curr_values[s] - values[s]))

            if delta <= self.tol * (1 - self.df) / self.df:
                if self.verbose:
                    print(f'Value iteration converged in {i+1} iterations')
                return values

        if self.verbose: print(f'Value iteration hits max iteration count {i}')
        return values

    def policy_eval(self, policy, values):
        if self.verbose: print(f'Policy eval: {policy}')
        for i in range(self.max_iter):
            for state in self.states:
                values[state] = self.R(state) + self.df * sum(
                    p * values[next_state] for p, next_state in self.T(
                        state, policy[state] if state in policy else None))
        return values

    def policy_iter(self):
        if self.verbose: print('\n==== MDP POLICY ITERATION ====')

        R, T, A = self.R, self.T, self.A

        values = {s: 0 for s in self.states}
        policy = {s: random.choice(A(s)) for s in self.transitions.keys()}
        for i in range(self.max_iter):
            values = self.policy_eval(policy, values)
            policy_stable = True

            for decision_node in self.transitions.keys():
                best_action = max(A(decision_node),
                                  key=lambda action: self.q_value(
                                      decision_node, action, values))
                if self.verbose:
                    print(f'Best action for {decision_node}: {best_action}')
                if self.q_value(
                        decision_node, best_action, values) > self.q_value(
                            decision_node, policy[decision_node], values):
                    policy[decision_node] = best_action
                    policy_stable = False

            if policy_stable:
                if self.verbose:
                    print(f'Policy iteration converged in {i+1} iterations')
                return policy, values


def print_dict(d, title=None):
    if title: print('==== ' + title + ' ====')
    for k in sorted(d.keys()):
        # print(f'{k}={v}')
        v = d[k]
        if type(v) == str: print('{:>10s}  ->  {:<10s}'.format(k, v))
        else: print('{:>10s}  =  {:<10.3f}'.format(k, v))
    print()


def main():
    args = get_args()
    nodes, edges, values, probs = parse_input(args.input_file)
    if args.v:
        print('==== PARSED INPUT ====')
        print('nodes', nodes)
        print('values', values)
        print('probs', probs)
        print('edges', edges)
        print()

    g = G(nodes,
          edges,
          values,
          probs,
          verbose=args.v,
          df=args.df,
          max_iter=args.iter,
          tol=args.tol)
    # print_dict(g.value_iter(), 'Value Iteration')
    p, v = g.policy_iter()
    if args.v:
        print()
    print_dict(p, 'BEST POLICY')
    print_dict(v, 'STATE VALUES')


def test2():
    args = get_args()
    # input_file = 'lab3/data/restaurant.txt'
    # input_file = 'lab3/data/publish.txt'
    # input_file = 'lab3/data/student.txt'
    input_file = 'lab3/data/student2.txt'
    nodes, edges, values, probs = parse_input(input_file)
    if args.v:
        print('nodes', nodes)
        print('values', values)
        print('probs', probs)
        print('edges', edges)
    print()
    g = G(nodes, edges, values, probs, verbose=True)
    print_dict(g.value_iter(), 'Value Iteration')
    p, v = g.policy_iter()
    print_dict(p, 'Best Policy')
    print_dict(v, 'Policy Iteration Values')
    # print(json.dumps(g.transitions, sort_keys=True))
    # print(json.dumps(g.value_iter(), sort_keys=True, indent=4))
    # print(json.dumps(p, sort_keys=True, indent=4))
    # print(json.dumps(v, sort_keys=True, indent=4))
    # ss = g.get_states_from_transitions(g.transitions)
    # print(ss)

    # mdp = MDP(nodes, edges, values, probs)


if __name__ == '__main__':
    main()
    # test()
    # test2()