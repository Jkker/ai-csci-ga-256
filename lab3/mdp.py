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
    parser.add_argument('-tol', type=float, default=0.01)
    parser.add_argument('-iter', type=int, default=100)
    parser.add_argument(
        'input_file',
        help='A mode-dependent input file',
        default='ex1.txt',
    )
    return parser.parse_args()


def parse_input(input_file, tol):
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

        for s in states:
            # for state, p in probs.items():
            if not s in probs and not s in edges and s in rewards:  # terminal node
                self.typeof[s] = 'terminal'
                if verbose: print(f'Terminal {s} = {rewards[s]}')
                continue

            if not s in probs:  # decision node w/o probs
                self.typeof[s] = 'decision'
                self.transitions[s] = dict((action, 1) for action in edges[s])
                if verbose: print(f'Decision w/o probs: {s}')
                continue

            actions = edges[s]
            if len(actions) == 1:  # decision node w/ only 1 neighbor
                self.typeof[s] = 'decision'
                self.transitions[s] = {actions[0]: 1}
                self.rewards[s] = rewards[s] if s in rewards else 0
                if verbose: print(f'Decision w/ only 1 neighbor: {s}')
                continue

            if len(probs[s]) == 1:  # decision node
                self.typeof[s] = 'decision'
                p_succ = probs[s][0]
                p_fail = round((1 - p_succ) / (len(actions) - 1), 3)
                self.transitions[s] = dict(
                    (action, [(p_succ if neighbor == action else p_fail,
                               neighbor) for neighbor in actions])
                    for action in actions)

                self.rewards[s] = rewards[s] if s in rewards else 0

                if verbose:
                    print(
                        f'Decision {s}: P = {p_succ}; R = {self.rewards[s]}; ACTIONS = {", ".join(actions)}'
                    )

            else:  # chance node
                self.typeof[s] = 'chance'
                self.probs[s] = dict(
                    (outcome, p) for outcome, p in zip(edges[s], probs[s]))

    def T(self, state, action=None):
        if self.typeof[state] == 'chance':
            return [(p, s) for s, p in self.probs[state].items()]
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
        R, T, A = self.R, self.T, self.A

        values = {s: 0 for s in self.states}
        policy = {s: random.choice(A(s)) for s in self.transitions.keys()}

        for i in range(self.max_iter):
            values = self.policy_eval(policy, values)
            policy_stable = True

            for state in self.transitions.keys():
                best_action = max(
                    A(state),
                    key=lambda action: self.q_value(state, action, values))
                if self.verbose:
                    print(f'Best action for {state}: {best_action}')
                if self.q_value(state, best_action, values) > self.q_value(
                        state, policy[state], values):
                    policy[state] = best_action
                    policy_stable = False

            if policy_stable:
                if self.verbose: print(f'Policy iteration converged in {i+1} iterations')
                return policy


class State:
    def __init__(self, name, value, actions, prob, tol=0.01):
        if value is None and not actions and not prob:
            raise ValueError(
                f'Node {name} must have one of the three entries to be valid')
        self.name = name
        self.value = value if value is not None else 0

        if not prob:
            self.type = 'terminal'
            self.prob = None
            self.actions = actions if actions else []
            return

        if len(prob) == 1:
            self.type = 'decision'
            self.prob = prob[0] if prob else 1
            self.actions = actions
            self.policy = None
            return

        if len(prob) > 1:
            self.type = 'chance'
            self.prob = prob
            if abs(sum(prob) - 1) > tol:
                raise ValueError('Probabilities must sum to 1.0')
            self.actions = actions
            return

    def print(self):
        if self.type == 'terminal':
            return f'{self.name} (T): {self.value}'
        elif self.type == 'decision':
            return f'{self.name} (D): {self.prob} -> {", ".join(self.actions)}'
        elif self.type == 'chance':
            d = ', '.join(
                [f'{n}={self.prob[i]}' for i, n in enumerate(self.actions)])
            return f'{self.name} (C): ({d})'

    def decision_probs(self, decision, alpha):
        return [(1 - alpha) if decision == a else alpha /
                len(self.actions[decision]) for a in self.actions[decision]]

    def eval(self, states, df, alpha):
        if self.type == 'terminal':
            return self.value
        elif self.type == 'decision':
            action_values = [states[a].value for a in self.actions]
            max_q = -float('inf')
            for d in self.actions:
                p = self.decision_probs(d, alpha)
                q = self.value + df * np.dot(p, action_values)
                if q > max_q:
                    max_q = q
                    self.policy = d
            return max_q

            # return max([self.prob * n.eval() for n in self.actions])
        elif self.type == 'chance':
            action_values = [states[a].value for a in self.actions]
            return self.value + df * np.dot(self.prob, action_values)


class MDP:
    policy = {}
    states = {}

    def __init__(self, states, edges, values, probs):
        for name in states:
            self.states[name] = State(name, values.get(name), edges.get(name),
                                      probs.get(name))

        for n in self.states.values():
            print(n.print())

    def eval(self, state):
        return self.states[state].eval()


def mps():
    policy = {}
    values = {}
    while True:
        values = value_iter(policy)
        new_policy = policy_iter(values)
        if new_policy == policy:
            break
        policy = new_policy
    return policy, values


def main():
    args = get_args()
    rewards, edges, probs = parse_input(args.input_file)
    print('rewards', rewards)
    print('edges', edges)
    print('probs', probs)
    policy, values = mps()
    print(policy)
    print(values)


import json


def test2():
    # args = get_args()
    input_file = 'lab3/data/publish.txt'
    tol = 0.01
    nodes, edges, values, probs = parse_input(input_file, tol)
    print('nodes', nodes)
    print('values', values)
    # print('edges', edges)
    print('probs', probs)
    print()
    g = G(nodes, edges, values, probs, verbose=True)
    # print(g.value_iter())
    # print(g.adj)
    print(json.dumps(g.transitions, sort_keys=True))
    print(json.dumps(g.value_iter(), sort_keys=True, indent=4))
    print(json.dumps(g.policy_iter(), sort_keys=True, indent=4))
    # ss = g.get_states_from_transitions(g.transitions)
    # print(ss)

    # mdp = MDP(nodes, edges, values, probs)


if __name__ == '__main__':
    # main()
    # test()
    test2()