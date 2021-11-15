import argparse
from copy import deepcopy
import random


# Argument & Input File Parser
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


# Helper Functions
def print_dict(d, title=None):
    if title: print('====== ' + title + ' ======')
    for k in sorted(d.keys()):
        v = d[k]
        if type(v) == str: print('{:>10s}  ->  {:<10s}'.format(k, v))
        else: print('{:>10s}  =  {:<10.3f}'.format(k, v))
    print()

class G:
    def __init__(self,
                 states,
                 edges,
                 rewards,
                 probs,
                 discount_factor=1.0,
                 tolerance=0.01,
                 max_iter=100,
                 minimize_rewards=False,
                 verbose=False):
        self.verbose = verbose
        self.df = discount_factor
        self.tol = tolerance
        self.max_iter = max_iter
        self.states = states
        self.optimize = min if minimize_rewards else max

        self.transitions = dict()
        self.rewards = rewards or {s: 0 for s in states}
        self.probs = dict()
        self.typeof = dict()
        if verbose: print('\n====== MDP INITIALIZATION ======')
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

    def T(self, state, action=None):  # get transition matrix
        if self.typeof[state] == 'chance':
            return self.probs[state]
        if self.typeof[state] == 'decision':
            return self.transitions[state][action]
        return [(0, state)]

    def A(self, state):  # get actions
        return list(self.transitions[state].keys()) if state in self.transitions else []

    def R(self, state):  # get rewards
        return self.rewards[state] if state in self.rewards else 0

    def bellman(self, state, action, values):  # use Bellman Equation to evaluate an action to a state
        if self.typeof[state] == 'terminal':
            return self.R(state)
        res = 0
        for p, next_state in self.T(state, action):
            res += p * (self.R(state) + self.df * values[next_state])
        return res

    def state_eval(self, state, values):  # compute value of a state
        if self.typeof[state] == 'terminal':
            return self.R(state)

        if self.typeof[state] == 'chance':
            return self.bellman(state, None, values)

        if self.typeof[state] == 'decision':
            return self.optimize(
                self.bellman(state, a, values) for a in self.A(state))

    # Value Iteration
    def value_iter(self):
        if self.verbose: print('\n====== MDP VALUE ITERATION ======')

        values = {s: 0 for s in self.states}
        for i in range(self.max_iter):
            curr_values = deepcopy(values)

            delta = 0

            for s in self.states:

                values[s] = self.state_eval(s, curr_values)

                delta = self.optimize(delta, abs(curr_values[s] - values[s]))

            if delta <= self.tol * (1 - self.df) / self.df:
                if self.verbose:
                    print(f'Value iteration converged in {i+1} iterations')
                return values

        if self.verbose: print(f'Value iteration hits max iteration count {i}')
        return values

    # Value computation of a given policy
    def policy_eval(self, policy, values):
        if self.verbose: print(f'Policy eval: {policy}')
        for i in range(self.max_iter):
            for state in self.states:
                values[state] = self.R(state) + self.df * sum(
                    p * values[next_state] for p, next_state in self.T(
                        state, policy[state] if state in policy else None))
        return values

    # Policy Iteration
    def policy_iter(self):
        if self.verbose: print('\n====== MDP POLICY ITERATION ======')

        q_value, A = self.bellman, self.A

        values = {s: 0 for s in self.states}
        policy = {s: random.choice(self.A(s)) for s in self.transitions.keys()}
        for i in range(self.max_iter):
            values = self.policy_eval(policy, values)
            policy_stable = True

            for decision_node in self.transitions.keys():
                best_action = self.optimize(self.A(decision_node),
                                            key=lambda action: q_value(
                                                decision_node, action, values))
                if self.verbose:
                    print(f'Best action for {decision_node}: {best_action}')

                if q_value(
                        decision_node, best_action, values) > q_value(
                            decision_node, policy[decision_node], values):
                    policy[decision_node] = best_action
                    policy_stable = False

            if policy_stable:
                if self.verbose:
                    print(f'Policy iteration converged in {i+1} iterations')
                return policy, values




def main():
    args = get_args()
    nodes, edges, rewards, probs = parse_input(args.input_file)
    if args.v:
        print('====== PARSED INPUT ======')
        print('NODES', nodes)
        print('REWARDS', rewards)
        print('PROBS', probs)
        print('EDGES', edges)
        print()

    g = G(nodes,
          edges,
          rewards,
          probs,
          verbose=args.v,
          discount_factor=args.df,
          max_iter=args.iter,
          minimize_rewards=args.min,
          tolerance=args.tol)
    # print_dict(g.value_iter(), 'Value Iteration')
    p, v = g.policy_iter()
    if args.v:
        print()
    print_dict(p, 'OPTIMAL POLICY')
    print_dict(v, 'STATE VALUES')

if __name__ == '__main__':
    main()