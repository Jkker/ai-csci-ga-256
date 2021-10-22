# CSCI-GA.2560-001 Lab 2

Author: Jerry Jia

Net-ID: `tj1043`

## Compile

It is not necessary to compile my code as it is written in Python.

## Running the Code

### For batch testing of the provided example 1 and 2

```sh
cd lab2/
./run.sh
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
