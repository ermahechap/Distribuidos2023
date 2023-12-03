# mpirun -np 4 encode.o -i "../Samples/custom_testcases/01.wav" -o "../Outputs/Audio/c_out.wav" -m "../MessageSamples/100000.txt" -t
mpirun -np 2 --hostfile gcp_hostfile encode.o -i "../Samples/custom_testcases/01.wav" -o "../Outputs/Audio/c_out.wav" -m "../MessageSamples/100.txt" -t
