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
        a[i] = atan2(data[i][1], data[i][0]);
    }
    return a;
}


char *stringToBinary(char* s) {
    if(s == NULL) return ""; /* no input string */
    int len = strlen(s);
    char *binary = malloc((len*8 + 1)); // each char is one byte (8 bits) and + 1 at the end for null terminator
    
    int binaryIndex = 0;
    for(int i = 0; i < len; ++i) {
        char currentChar = s[i];
        for(int j = 7; j >= 0; j--){
            binary[binaryIndex++] = ((currentChar >> j) & 1) ? '1' : '0';
        }
    }
    binary[binaryIndex] = '\0';
    return binary;
}

char* binaryToString(char *binary) {
    int binaryLength = strlen(binary);
    int stringLength = binaryLength / 8;
    char* originalString = malloc(stringLength + 1);

    int binaryIndex = 0;
    for (int i = 0; i < stringLength; i++) {
        char currentChar = 0;
        for (int j = 0; j < 8; j++) {
            currentChar = (currentChar << 1) | (binary[binaryIndex++] - '0');
        }
        originalString[i] = currentChar;
    }
    originalString[stringLength] = '\0';
    return originalString;
}

int main(void) {
    // ------------------------ Setup --------------------
    char in_filename[] = "../../Outputs/c_out.wav"; // Output filename 
    
    // ---------------------- Load WAV -------------------
    SNDFILE *infile;
    SF_INFO sfinfo_in;
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
    for (int i = (n_frames/2)+1; i < n_frames; i++) {
        out[i][0] = out[n_frames - i][0];
        out[i][1] = -out[n_frames - i][1];
    }

    fftw_complex *shifted = ffshift(out, n_frames);


    int frame = 168;
    int embed_sample_sz = 10;
    int p = frame * embed_sample_sz; // total number of samples used for embedding data
    int centre = n_frames / 2 + 1; // centre frequency/ zero point
    int embedding_freq = 5000; // in hz

    printf("p = %d\n", p);
    printf("centre = %d\n", centre);
    printf("embedding_freq = %d\n", embedding_freq);

    double *X_abs = magnitude(shifted, n_frames);
    double *X_angle = angle(shifted, n_frames);

    FILE *fft_abs_altered = fopen("../Testing/c_fft_magnitude_decoded.txt", "w");
    for (int i = 0; i < n_frames; i++) {
        fprintf(fft_abs_altered, "%f\t%f\n", X_abs[i], X_angle[i]);
    }
    fclose(fft_abs_altered);

    int start_embed = centre - embedding_freq - p;
    int end_embed = centre - embedding_freq;
    double *X_embed = malloc(sizeof(double) * p);
    
    memcpy(X_embed, &X_abs[start_embed], sizeof(double) * p);
    // for (int i = 0; i < p; i++){
    //     X_embed[i] = X_abs[i + start_embed];
    // }

    double a = 0.1; // amplification factor of embedding

    char *recovered_binary = malloc(frame * sizeof(char));
    // Decode loop
    for (int k = 0; k < frame; k++){
        double avg = 0;
        int b = 0, c = 0;
        for (int l = 0; l < embed_sample_sz; l++) {
            avg = avg + X_embed[k * embed_sample_sz + l];
        }
        avg = avg / embed_sample_sz;

        for (int l = 0; l < embed_sample_sz / 2; l++){
            if (X_embed[k * embed_sample_sz +l] >= (1 + a) * avg / 2){
                c++;
            } else {
                b++;
            }
        }

        for (int l = embed_sample_sz / 2; l < embed_sample_sz - 1; l++){
            if (X_embed[k * embed_sample_sz + l] < (3 - a) * avg / 2){
                c++;
            } else {
                b++;
            }
        }
        printf("%d - %d\n", b, c);
        recovered_binary[k] = (b > c) ? '1': '0';
    }
    printf("Recovered Encoded Message: %s", recovered_binary);
}