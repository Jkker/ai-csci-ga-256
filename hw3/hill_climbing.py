import heapq
from copy import deepcopy


class Knapsack:
    def __init__(
        self,
        state,
        objects,
        max_weight,
        target_value,
    ) -> None:
        self.state = set(state)
        self.objects = objects
        self.max_weight = max_weight
        self.target_value = target_value

    def rest_objects(self):
        return set(self.objects.keys()) - self.state

    def value(self):
        return sum([self.objects[o][0] for o in self.state])

    def weight(self):
        return sum([self.objects[o][1] for o in self.state])

    def add(self, new_object):
        self.state.add(new_object)

    def delete(self, new_object):
        self.state.remove(new_object)

    def __str__(self) -> str:
        return '{' + ','.join(self.state) + '}'

    def __lt__(self, other):
        e1 = self.error()
        e2 = other.error()
        if e1 < e2:
            return True
        elif e1 > e2:
            return False
        else:  # equal cost
            v1 = self.value()
            v2 = other.value()
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
            else:  # equal value
                return self.weight() < other.weight()

    def __le__(self, other):
        return self.error() <= other.error()

    def error(self):
        return max(self.weight() - self.max_weight, 0) + max(
            self.target_value - self.value(), 0)

    def hill_climbing(self):
        state = self
        while not state.error() == 0:
            print(f'🥡 Current State = {state}')
            neighbors = []

            # add
            for o in state.rest_objects():
                new_state = deepcopy(state)
                new_state.add(o)
                heapq.heappush(neighbors, (new_state, f'➕  Add {o}  '))

            # delete
            for o in state.state:
                new_state = deepcopy(state)
                new_state.delete(o)
                heapq.heappush(neighbors, (new_state, f'➖  Del {o}  '))

            # swap
            for o1 in state.state:
                for o2 in state.rest_objects():
                    new_state = deepcopy(state)
                    new_state.delete(o1)
                    new_state.add(o2)
                    heapq.heappush(neighbors, (new_state, f'🔄 Swap {o1} {o2}'))

            for n in neighbors:
                print(f'{n[1]}: h({n[0]}) = {n[0].error()}')

            if neighbors[0][0] > state:
                raise Exception("Unable to improve by %s" % neighbors[0][0])
            state = neighbors[0][0]
            print(f'🌟 New State = {state}\n')

        return state


if __name__ == '__main__':
    objects = {
        'A': (10, 8),
        'B': (8, 4),
        'C': (7, 3),
        'D': (6, 3),
        'E': (4, 1)
    }

    curr = ['A', 'E']

    print(f'🎯 Solution: {Knapsack(curr, objects, 10, 20).hill_climbing()}')
