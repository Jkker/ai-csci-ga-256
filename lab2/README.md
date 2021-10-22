# CSCI-GA.2560-001 Lab 2

Author: Jerry Jia

Net-ID: `tj1043`

## Compile

It is not necessary to compile my code as it is written in Python.

## Running the Code

### For batch testing with example 1 and 2

```sh
cd lab2/
./run.sh
```

#### Expected Output

> Note: DPLL output is NOT consistent due to random selection of unassigned literal in a hard case

```sh
> python3 main.py data/BNF_ex2.txt -mode cnf
!C !B A
!C !A B
!B !A C
A B C
A !B
> python3 main.py data/CNF_ex1.txt -mode dpll
A = False
B = True
C = False
P = False
Q = True
W = False
> python3 main.py data/CNF_ex2.txt -mode dpll
A = True
B = False
C = True
> python3 main.py data/BNF_ex1.txt -mode solver
B = True
C = True
P = True
Q = False
W = False
A = False
> python3 main.py data/BNF_ex2.txt -mode solver
C = True
B = False
A = True
```

### Using CLI

```sh
python3 main.py data/BNF_ex1.txt -mode cnf -v
python3 main.py data/BNF_ex2.txt -mode cnf -v

python3 main.py data/CNF_ex1.txt -mode dpll -v
python3 main.py data/CNF_ex2.txt -mode dpll -v

python3 main.py data/BNF_ex1.txt -mode solver -v
python3 main.py data/BNF_ex2.txt -mode solver -v
```

## Environment

```sh
❯ python -V
Python 3.9.5
```
