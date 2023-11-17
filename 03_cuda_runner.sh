#!/bin/bash
n_iters=1
audio_durations=("01" "05" "10" "20" "30" "40")
msg_lengths=("100" "1000" "10000" "100000" "1000000")
bucket_threads=(16 32 64 128 256 512)

for duration in "${audio_durations[@]}"; do
    for len in "${msg_lengths[@]}"; do
        for thread in "${bucket_threads[@]}"; do
            for ((i=1; i<=n_iters; i++)); do
                echo -n "duration: ${duration}, msg_len: ${len}, n_threads: ${thread}, iteration: ${i}, " &&
                ./"03 Cuda/watermarking" -i "./Samples/custom_testcases/${duration}.wav" -o "./Outputs/Audio/c_out.wav" -m "./MessageSamples/${len}.txt" -n ${thread} -t
                sleep 2
            done
        done
    done
done
