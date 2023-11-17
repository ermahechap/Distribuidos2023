#!/bin/bash
./01_normal_runner.sh >> "00 Results/01_normal_results.txt"
./02_OMP_runner.sh >> "00 Results/02_OMP_results.txt"
./03_cuda_runner.sh >> "00 Results/03_cuda_results.txt"
