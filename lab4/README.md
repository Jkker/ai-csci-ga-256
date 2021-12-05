# CSCI-GA.2560-001 Lab 4

Author: Jerry Jia

Net-ID: `tj1043`

## Installation

### Environment

```sh
❯ python -V
Python 3.9.5
```

### Using `pip`

```sh
pip install numpy pandas
```

### Using `conda`

```sh
conda install numpy pandas
```

## Running KNN

### Batch run of all given KNN examples

```sh
cd lab4/
chmod +x run_knn.sh
./run_knn.sh
```

### Batch testing KNN

`test_knn.py` executes knn for all test cases and compares them with the provided outputs to evaluate a passing rate.

```sh
cd lab4/
python3 test_knn.py
```

Note: add `-v` flag for verbose output

### Running a single KNN example

```sh
python3 knn.py -k 3 -d e2 -train data/knn1.train.txt -test data/knn1.test.txt
```

## Running KMeans

### Batch run of all given KMeans examples

```sh
cd lab4/
chmod +x run_kmeans.sh
./run_kmeans.sh
```

### Running a single KMeans example

```sh
python3 kmeans.py -d e2 -data data/km1.txt 0,0 200,200 500,500
```
