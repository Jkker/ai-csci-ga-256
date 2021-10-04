## AI HW3: Knapsack Hill Climbing Local Search

### 1. [5 pts.] Suppose at some iteration of simple hill climbing the current state is {A,E}.

#### What is the best neighbor of the state {A,E}?

```
🥡 Current State = {A,E}
➕ Add C  : h({A,C,E}) = 2  # ✅ best neighbor
➕ Add D  : h({A,D,E}) = 2
➕ Add B  : h({A,B,E}) = 3
🔄 Swap E C: h({A,C}) = 4
🔄 Swap E B: h({A,B}) = 4
🔄 Swap A C: h({C,E}) = 9
🔄 Swap A D: h({D,E}) = 10
➖ Del A  : h({E}) = 16
🔄 Swap A B: h({B,E}) = 8
➖ Del E  : h({A}) = 10
🔄 Swap E D: h({A,D}) = 5
🌟 New State = {A,C,E}
```

#### What happens on the next iteration?

```
🥡 Current State = {A,C,E}
🔄 Swap A B: h({C,E,B}) = 1  # ✅ best neighbor
🔄 Swap C D: h({A,E,D}) = 2
🔄 Swap A D: h({C,E,D}) = 3
🔄 Swap C B: h({A,E,B}) = 3
🔄 Swap E D: h({A,C,D}) = 4
➖ Del A  : h({C,E}) = 9
➖ Del E  : h({A,C}) = 4
➖ Del C  : h({A,E}) = 6
➕ Add D  : h({A,C,E,D}) = 5
➕ Add B  : h({A,C,E,B}) = 6
🔄 Swap E B: h({A,C,B}) = 5
🌟 New State = {C,E,B}

🥡 Current State = {C,E,B}
🔄 Swap E D: h({C,B,D}) = 0  # ✅ best neighbor
➕ Add D  : h({C,B,E,D}) = 1
🔄 Swap C D: h({B,E,D}) = 2
🔄 Swap E A: h({C,B,A}) = 5
🔄 Swap B A: h({C,E,A}) = 2
➖ Del C  : h({B,E}) = 8
🔄 Swap C A: h({B,E,A}) = 3
➕ Add A  : h({C,B,E,A}) = 6
➖ Del E  : h({C,B}) = 5
➖ Del B  : h({C,E}) = 9
🔄 Swap B D: h({C,E,D}) = 3
🌟 New State = {C,B,D}

🎯 Solution: {C,B,D}
```

### 2. [5 pts.] Consider now the general case where there are $N$ objects.

#### What is the size of the state space?

$$\text{size of state space} = \sum_{k=1}^N C(N,k)=\sum_{k=1}^N \frac{N!}{k!(N-k)!}=\boxed{2^N-1}$$

#### What is maximal number of neighbors of any state?

Let $x$ be the number of objects in the knapsack

$$\begin{align*} {n_\text{neighbors}} &=  n_\text{add} + n_\text{delete} + n_\text{swap} \\&=(N-x)+x+x(N-x) \\&=N+xN-x^2 \end{align*}$$

$$\max\{n_\text{neighbors}\} = \max\{N+xN-x^2\} = \boxed{\frac{N(N+4)}{4}}\quad\text{at}\quad x=\frac{N}{2}$$