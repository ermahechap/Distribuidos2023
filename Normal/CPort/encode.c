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


fftw_complex *iffshift(fftw_complex *data, int n){
    fftw_complex *unshifted = fftw_malloc(sizeof(fftw_complex) * n);
    if (n % 2 == 0) {
        return ffshift(data, n);
    }
    
    int shift = (n - 1) / 2;
    for(int i = 0; i < n; i++) {
        int j = i + shift + 1;
        unshifted[i][0] = data[j][0];
        unshifted[i][1] = data[j][1];
    }
    return unshifted;
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
        a[i] = atan2(data[i][1], data[i][0]);
    }
    return a;
}


char *stringToBinary(char* s) {
    if(s == NULL) return ""; /* no input string */
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

char *binaryToString(char *b) {
    size_t len = strlen(b) / 8;
    char *str = malloc(len*sizeof(char));
    printf("------------\n");
    for(size_t i = 0; i < strlen(b); i+=8) {
        char ch = 0;
        for (int j = 0; j <= 7; j++) {
            int bit = (b[i + j] == '1') ? 1: 0;
            ch += bit << 7 - j;
        }
        str[i / 8] = ch;
    }
    return str;
}

int main(void) {
    // ------------------------ Setup --------------------
    char in_filename[] = "../../Samples/ImperialMarch60.wav"; // Input filename
    char out_filename[] = "../../Outputs/c_out.wav"; // Output filename
    char message[] = "My name is Slim Shady"; //Message 
    
    // ---------------------- Load WAV -------------------
    SNDFILE *infile, *outfile;
    SF_INFO sfinfo_in, sfinfo_out;
    double *buffer;
    double *data;

    infile = sf_open(in_filename, SFM_READ, &sfinfo_in);
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
    printf("-----------------\n");

    // --------------------- FFT ------------------
    // FFT
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n_frames);
    fftw_plan plan = fftw_plan_dft_r2c_1d(n_frames, data, out, FFTW_ESTIMATE); // Create fftw execution plan
    fftw_execute(plan); // Execute fftw plan

    // ffshift
    fftw_complex *shifted = ffshift(out, n_frames);

    printf("----------------\n");
    for(int i = n_frames/2; i < n_frames/2+10; i++){
        printf("%.3f + %.3fj\n", shifted[i][0], shifted[i][1]);
    }
    // ------------- Embed meesage ----------------
    // To binary
    char *binary_msg = stringToBinary(message);
    printf("Message n_bits: %d\n", strlen(binary_msg));
    printf("Message: %s\n", message);
    printf("Encoded Message: %s\n", binary_msg);

    // Embedding algorithm
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
            avg += X_embed[k * embed_sample_sz + l];
        }
        avg /= embed_sample_sz;

        if (binary_msg[k] == '0') {
            // printf("0 ");
            for (int l = 0; l < embed_sample_sz; l++){
                X_embed[k*embed_sample_sz + l] = avg;
            }
        } else {
            // printf("1 ");
            for (int l = 0; l < embed_sample_sz/2; l++){
                X_embed[k*embed_sample_sz + l] = a*avg;
            }
            for (int l = embed_sample_sz/2; l < embed_sample_sz; l++){
                X_embed[k*embed_sample_sz + l] = (2-a)*avg;
            }
        }
    }

    // define range for adding embeddings back to final fft vec with embeddings
    // centre - embedding_freq - p <--- centre ---> centre + embedding_freq + p

    int range_1[] = {centre - embedding_freq - p, centre - embedding_freq}; // [centre - freq - p, centre - freq]
    int range_2[] = {centre + embedding_freq + 1, centre + embedding_freq + p + 1}; // [centre + freq, centre + freq + p]

    for (int i = range_1[0]; i < range_1[1]; i++){
        X_abs[i] = X_embed[i - range_1[0]];
    }

    //symmetry
    for (int i = range_2[0]; i < range_2[1]; i++){
        X_abs[i] = X_embed[range_2[1] - i];
    }

    // multiply
    fftw_complex *final = fftw_malloc(sizeof(fftw_complex)* n_frames);
    for (int i = 0; i < n_frames; i++) {
        final[i][0] = X_abs[i] * cos(X_angle[i]);
        final[i][1] = X_abs[i] * sin(X_angle[i]);
    }


    printf("----------------\n");
    for(int i = n_frames/2; i < n_frames/2+10; i++){
        printf("%.3f + %.3fj\n", final[i][0], final[i][1]);
    }

    // -------------------- Write WAV --------------------
    // Unshift
    fftw_complex *unshifted = iffshift(final, n_frames);
    printf("----------------\n");
    for(int i = 0; i < 10; i++){
        printf("%.3f + %.3fj\n", unshifted[i][0], unshifted[i][1]);
        //printf("%.3f + %.3fj\n", out[i][0], out[i][1]);
    }
    // IFFT
    double *restored_signal = malloc(sizeof(double) * n_frames);
    plan = fftw_plan_dft_c2r_1d(n_frames, unshifted, restored_signal, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    // Normalization
    for (int i = 0; i < n_frames; i++) {
        restored_signal[i];
    }

    printf("--------------\n");
    for (int i = 0; i< n_frames; i++){
        if(restored_signal[i] != data[i]) {
            printf("%d >>> %d (%d)\n", i, restored_signal[i], data[i]);
        }
    }

    sfinfo_out.samplerate = fs;
    sfinfo_out.channels = 1;  // Writing only the first channel
    sfinfo_out.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE; // WAV format, floating-point samples

    outfile = sf_open(out_filename, SFM_WRITE, &sfinfo_out);
    sf_write_double(outfile, restored_signal, n_frames);

    sf_close(outfile);
    free(buffer);
    free(data);
}
