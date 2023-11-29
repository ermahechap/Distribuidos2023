#!/bin/bash
n_iters=5
audio_durations=("01" "05" "10" "20" "30" "40")
msg_lengths=("100" "1000" "10000" "100000" "1000000")
threads=(1 2 4 8 16)

for duration in "${audio_durations[@]}"; do
    for len in "${msg_lengths[@]}"; do
        for thread in "${threads[@]}"; do
            for ((i=1; i<=n_iters; i++)); do
                echo -n "duration: ${duration}, msg_len: ${len}, n_threads: ${thread}, iteration: ${i}, " &&
                ./"02 OMP/encode.o" -i "./Samples/custom_testcases/${duration}.wav" -o "./Outputs/Audio/c_out.wav" -m "./MessageSamples/${len}.txt" -n ${thread} -t
                sleep 1
            done
        done
    done
done
