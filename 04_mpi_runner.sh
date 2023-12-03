#!/bin/bash
hostfile=gcp_hostfile
n_iters=5
audio_durations=("01" "05" "10" "20" "30" "40")
msg_lengths=("100" "1000" "10000" "100000" "1000000")
number_of_ranks=(1 2 4 8)

for duration in "${audio_durations[@]}"; do
    for len in "${msg_lengths[@]}"; do
        for n_ranks in "${number_of_ranks[@]}"; do
            for ((i=1; i<=n_iters; i++)); do
                echo -n "duration: ${duration}, msg_len: ${len}, n_ranks: ${n_ranks}, iteration: ${i}, " &&
                mpirun -np ${n_ranks} --hostfile "${hostfile}" "04 MPI/encode.o" -i "./Samples/custom_testcases/${duration}.wav" -o "./Outputs/Audio/c_out.wav" -m "./MessageSamples/${len}.txt" -t
                sleep 1
            done
        done
    done
done
