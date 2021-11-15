#!/bin/bash

echo "> python3 mdp.py data/maze.txt"
python3 mdp.py data/maze.txt

echo "> python3 mdp.py data/publish.txt"
python3 mdp.py data/publish.txt

echo "> python3 mdp.py data/restaurant.txt -min"
python3 mdp.py data/restaurant.txt -min

echo "> python3 mdp.py data/student.txt"
python3 mdp.py data/student.txt

echo "> python3 mdp.py data/student2.txt"
python3 mdp.py data/student2.txt

echo "> python3 mdp.py data/teach.txt -df 0.9"
python3 mdp.py data/teach.txt -df 0.9