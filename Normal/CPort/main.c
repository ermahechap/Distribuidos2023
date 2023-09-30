#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sndfile.h>
#include <fftw3.h>
#include <complex.h>

double *readWav(char *path, SF_INFO sfInfo, SNDFILE *sfFile) {
    sfFile = sf_open(path, SFM_READ, &sfInfo);
    double *data = malloc(sizeof(double) * sfInfo.frames * sfInfo.channels);
    sf_read_double(sfFile, data, sfInfo.frames * sfInfo.channels);

    return data;
}

void writeWav(double *data, char *path, SF_INFO sfInfo, SNDFILE *sfFile) {
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE; // WAV format, floating-point samples
    sfFile = sf_open(path, SFM_WRITE, &sfInfo);
    sf_write_double(sfFile, data, sfInfo.frames * sfInfo.channels);
}

fftw_complex *ffshift(fftw_complex *data, int n) {
    fftw_complex *shifted = fftw_malloc(sizeof(fftw_complex) * n);
    
    int shift = n / 2;
    for (int i = 0; i < n; i++){
        int j = (i + shift) % n;
        shifted[i][0] = data[j][0];
        shifted[i][1] = data[j][1];
    }
    return shifted;
}

double *magnitude(fftw_complex *data, int n) {
    double *m = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        m[i] = sqrt(data[i][0]*data[i][0] + data[i][1]*data[i][1]);
    }
    return m;
}

double *angle(fftw_complex *data, int n) {
    double *a = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = atan(data[i][1] / data[i][0]);
    }
}


char* stringToBinary(char* s) {
    if(s == NULL) return 0; /* no input string */
    size_t len = strlen(s);
    char *binary = malloc(len*8 + 1); // each char is one byte (8 bits) and + 1 at the end for null terminator
    binary[0] = '\0';
    for(size_t i = 0; i < len; ++i) {
        char ch = s[i];
        for(int j = 7; j >= 0; --j){
            if(ch & (1 << j)) {
                strcat(binary,"1");
            } else {
                strcat(binary,"0");
            }
        }
    }
    return binary;
}

int main(void) {
    char message[] = "This is an embedded message";
    // ---------------------- Load WAV -------------------
    char in_filename[] = "../sample.wav";
    char out_filename[] = "../c_out.wav";

    SNDFILE *infile, *outfile;
    SF_INFO sfinfo_in, sfinfo_out;
    double *buffer;
    double *data;

    infile = sf_open("../sample.wav", SFM_READ, &sfinfo_in);
    buffer = (double *)malloc(sfinfo_in.channels * sfinfo_in.frames * sizeof(double));
    data = (double *)malloc(sfinfo_in.frames * sizeof(double));
    sf_read_double(infile, buffer, sfinfo_in.channels * sfinfo_in.frames);

    if (sfinfo_in.channels > 1){
        printf("WARNING - More than one channel, taking first only!\n");
        for (int i = 0; i < sfinfo_in.frames; i++) {
            data[i] = buffer[i * sfinfo_in.channels];
        }
    } else {
        memcpy(data, buffer, sizeof(double) * sfinfo_in.frames * sfinfo_in.channels);
    }

    sf_close(infile);

    // file info
    int fs = sfinfo_in.samplerate; // wav sample rate
    int n_frames = sfinfo_in.frames; // wav frames
    int n_channels = sfinfo_in.channels; // wav channels
    printf("Input file path: %s\n", in_filename);
    printf("Sample Rate: %d Hz\n", fs);
    printf("Frames: %d\n", n_frames);
    printf("Channels: %d\n", n_channels);

    // --------------------- FFT ------------------
    // FFT
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n_frames);
    fftw_plan plan = fftw_plan_dft_r2c_1d(n_frames, data, out, FFTW_ESTIMATE); // Create fftw execution plan
    fftw_execute(plan); // Execute fftw plan


    // TODO, REMOVE... Print to file to validate fft results
    FILE *fptr;
    fptr = fopen("./test_out/1_fft.txt", "w");
    for (int i = 0; i < n_frames; i++) {
        fprintf(fptr, "%f %f\n", out[i][0], out[i][1]);
    }
    fclose(fptr);


    // ffshift
    fftw_complex *shifted = ffshift(out, n_frames);
    // TODO, REMOVE... Print to file to validate fft results
    fptr = fopen("./test_out/2_fft_shifted.txt", "w");
    for (int i = 0; i < n_frames; i++) {
        fprintf(fptr, "%f %f\n", shifted[i][0], shifted[i][1]);
    }
    fclose(fptr);


    // ------------------ Embed msg ----------------
    // To binary
    char *binary_msg = stringToBinary(message);

    // Embedding algorithm
    //int frame = sizeof(binary_msg) / sizeof(char);
    int frame = strlen(binary_msg);
    int embed_sample_sz = 10;
    int p = frame * embed_sample_sz; // total number of samples used for embedding data
    int centre = n_frames / 2 + 1; // centre frequency/ zero point
    int embedding_freq = 5000; // in hz

    double *X_abs = magnitude(shifted, n_frames);
    double *X_angle = angle(shifted, n_frames);

    int start_embed = centre - embedding_freq - p;
    int end_embed = centre - embedding_freq;
    double *X_embed = malloc(sizeof(double) * p);
    memcpy(X_embed, X_abs+start_embed, p);

    double a = 0.1; // amplification factor of embedding

    // Loop
    for (int k = 0; k < frame; k++) { // row
        double avg = 0;
        for (int l = 0; l < embed_sample_sz; l++){ //col
            avg = avg + X_embed[k*embed_sample_sz + l];
        }
        avg = avg / embed_sample_sz;

        if (binary_msg[k] == '0') {
            printf("0 ");
            for (int l = 0; l < embed_sample_sz; l++){
                X_embed[k*embed_sample_sz + l] = avg;
            }
        } else {
            printf("1 ");
            for (int l = 0; l < embed_sample_sz/2; l++){
                X_embed[k*embed_sample_sz + l] = a*avg;
            }
            for (int l = 0; l < embed_sample_sz/2; l++){
                X_embed[k*embed_sample_sz + l] = (2-a)*avg;
            }
        }
    }

    // define range for adding embeddings back to final fft vec with embeddings
    int range_1[] = {centre-embedding_freq-p, centre-embedding_freq};
    int range_2[] = {centre+embedding_freq+1, centre+embedding_freq+p+1};

    for (int i = range_1[0]; i < range_1[1]; i++){
        X_abs[i] = X_embed[i - range_1[0]];
    }

    //symmetry
    for (int i = range_2[0]; i < range_2[1]; i++){
        X_abs[i] = X_embed[range_2[0] - i];
    }
    // -------------------- Write WAV --------------------

    // IFFT
    double *restored_signal = malloc(sizeof(double) * n_frames);
    plan = fftw_plan_dft_c2r_1d(n_frames, out, restored_signal, FFTW_ESTIMATE);
    fftw_execute(plan);
    // Normalization
    for (int i = 0; i < n_frames; i++) {
        restored_signal[i] /= n_frames;

    }

    sfinfo_out.samplerate = fs;
    sfinfo_out.channels = 1;  // Writing only the first channel
    sfinfo_out.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE; // WAV format, floating-point samples

    outfile = sf_open("../c_out.wav", SFM_WRITE, &sfinfo_out);
    sf_write_double(outfile, restored_signal, n_frames);

    sf_close(outfile);
    free(buffer);
    free(data);


}
