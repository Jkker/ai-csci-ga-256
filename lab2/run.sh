#!/bin/bash

echo "> python3 main.py data/BNF_ex1.txt -mode cnf -v"
python3 main.py data/BNF_ex1.txt -mode cnf -v
echo "> python3 main.py data/BNF_ex2.txt -mode cnf -v"
python3 main.py data/BNF_ex2.txt -mode cnf -v

echo "> python3 main.py data/CNF_ex1.txt -mode dpll -v"
python3 main.py data/CNF_ex1.txt -mode dpll -v
echo "> python3 main.py data/CNF_ex2.txt -mode dpll -v"
python3 main.py data/CNF_ex2.txt -mode dpll -v

echo "> python3 main.py data/BNF_ex1.txt -mode solver -v"
python3 main.py data/BNF_ex1.txt -mode solver -v
echo "> python3 main.py data/BNF_ex2.txt -mode solver -v"
python3 main.py data/BNF_ex2.txt -mode solver -v