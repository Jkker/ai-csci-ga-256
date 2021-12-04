#!/bin/bash

printf "> python3 kmeans.py -d e2 -data data/km1.txt 0,0 200,200 500,500\n"
python3 kmeans.py -d e2 -data data/km1.txt 0,0 200,200 500,500

printf "> python3 kmeans.py -d manh -data data/km1.txt 0,0 200,200 500,500\n"
python3 kmeans.py -d manh -data data/km1.txt 0,0 200,200 500,500

printf "> python3 kmeans.py -d e2 -data data/km2.txt 0,0,0 200,200,200 500,500,500\n"
python3 kmeans.py -d e2 -data data/km2.txt 0,0,0 200,200,200 500,500,500

printf "> python3 kmeans.py -d manh -data data/km2.txt 0,0,0 200,200,200 500,500,500\n"
python3 kmeans.py -d manh -data data/km2.txt 0,0,0 200,200,200 500,500,500