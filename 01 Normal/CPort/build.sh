gcc encode.c -o encode.o -lsndfile -lfftw3 -lm  &&
gcc decode.c -o decode.o -lsndfile -lfftw3 -lm