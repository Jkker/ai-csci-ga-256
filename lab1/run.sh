#!/bin/bash

echo "> python3 main.py ex1.txt -start S -end G -alg BFS -v"
python3 main.py ex1.txt -start S -end G -alg BFS -v
echo "> python3 main.py ex1.txt -start S -end G -alg ID -depth 2 -v"
python3 main.py ex1.txt -start S -end G -alg ID -depth 2 -v
echo "> python3 main.py ex1.txt -start S -end G -alg ASTAR -v"
python3 main.py ex1.txt -start S -end G -alg ASTAR -v
echo "> python3 main.py ex1.txt -start S -end G -alg ASTAR -v"
python3 main.py ex2.txt -start S -end G -alg BFS -v
echo "> python3 main.py ex2.txt -start S -end G -alg ID -depth 2 -v"
python3 main.py ex2.txt -start S -end G -alg ID -depth 2 -v
echo "> python3 main.py ex2.txt -start S -end G -alg ASTAR -v"
python3 main.py ex2.txt -start S -end G -alg ASTAR -v