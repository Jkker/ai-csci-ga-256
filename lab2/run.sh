#!/bin/bash

echo "> python3 main.py data/BNF_ex1.txt -mode cnf"
python3 main.py data/BNF_ex1.txt -mode cnf
echo "> python3 main.py data/BNF_ex2.txt -mode cnf"
python3 main.py data/BNF_ex2.txt -mode cnf

echo "> python3 main.py data/CNF_ex1.txt -mode dpll"
python3 main.py data/CNF_ex1.txt -mode dpll
echo "> python3 main.py data/CNF_ex2.txt -mode dpll"
python3 main.py data/CNF_ex2.txt -mode dpll

echo "> python3 main.py data/BNF_ex1.txt -mode solver"
python3 main.py data/BNF_ex1.txt -mode solver
echo "> python3 main.py data/BNF_ex2.txt -mode solver"
python3 main.py data/BNF_ex2.txt -mode solver