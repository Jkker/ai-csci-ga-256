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

```sh
> python3 main.py data/BNF_ex1.txt -mode cnf
!A !P Q W
!B !C !P Q W
!W A B
!W A C
P A B
P A C
!Q A B
!Q A C
!A B
C B
C A
> python3 main.py data/BNF_ex2.txt -mode cnf
!C !A B
!C !B A
A B C
!B !A C
A !B
> python3 main.py data/CNF_ex1.txt -mode dpll
A = True
B = True
C = False
P = False
❯ ./run.sh
> python3 main.py data/BNF_ex1.txt -mode cnf
!A !P Q W
!B !C !P Q W
!W A B
!W A C
P A B
P A C
!Q A B
!Q A C
!A B
C B
C A
> python3 main.py data/BNF_ex2.txt -mode cnf
!C !A B
!C !B A
A B C
!B !A C
A !B
> python3 main.py data/CNF_ex1.txt -mode dpll
A = True
B = True
C = False
P = False
Q = False
W = False
> python3 main.py data/CNF_ex2.txt -mode dpll
A = True
B = True
C = True
> python3 main.py data/CNF_ex3.txt -mode dpll
P = False
Q = True
R = False
U = True
W = False
X = False
> python3 main.py data/CNF_ex4.txt -mode dpll
P = False
Q = False
R = False
W = False
X = False
> python3 main.py data/BNF_ex1.txt -mode solver
A = True
B = True
C = False
P = False
Q = False
W = False
> python3 main.py data/BNF_ex2.txt -mode solver
A = True
B = True
C = True
```

### Using CLI

```sh
python3 main.py data/BNF_ex1.txt -mode cnf -v
python3 main.py data/BNF_ex2.txt -mode cnf -v

python3 main.py data/CNF_ex1.txt -mode dpll -v
python3 main.py data/CNF_ex2.txt -mode dpll -v
python3 main.py data/CNF_ex3.txt -mode dpll -v
python3 main.py data/CNF_ex4.txt -mode dpll -v

python3 main.py data/BNF_ex1.txt -mode solver -v
python3 main.py data/BNF_ex2.txt -mode solver -v
```

## Environment

```sh
❯ python -V
Python 3.9.5
```
