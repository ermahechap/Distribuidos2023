#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sndfile.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>

char *stringToBinary(char* s) {
    if(s == NULL) return ""; /* no input string */
    size_t len = strlen(s);
    char *binary = (char *)malloc(len*8 + 1); // each char is one byte (8 bits) and + 1 at the end for null terminator
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
    char *str = (char *)malloc(len*sizeof(char));
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

__global__ void fftshift(cufftComplex *data, int n_frames) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n_frames / 2) {
        cufftComplex temp = data[id]; // first half value to temp
        data[id] = data[id + n_frames / 2]; // update first half position /w second half value
        data[id + n_frames / 2] = temp; // update second half with temp (copied value)
    }
}

__global__ void getMagnitudeAngle(cufftComplex *data, float *magnitude, float *angle, int n_samples) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n_samples) {
        magnitude[id] = sqrt(data[id].x * data[id].x) * (data[id].y +data[id].y);
        angle[id] = atan2(data[id].y, data[id].x);
    };
}


// __global__ void embeddingLoop(float *X_embed, char *msg, int frame, int embed_sample_sz, double a) {
//     __shared__ float X_embed_temp[2 * embed_sample_sz];
//     int k = threadIdx.x + blockIdx.x * blockDim.x;

//     // Read into shared X_embed:
//     X_embed_temp[k] = X_embed[k * embed_sample_sz];
//     if (threadIdx.x < 2 * embed_sample_sz) {
//         X_embed_temp[k + embed_sample_sz] = X_embed[k*embed_sample_sz + threadIdx.x];
//     }

//     if (k < frame) {
//         float avg = 0;
//         for (int l = 0; l < embed_sample_sz; l++){
//             avg += X_embed[k * embed_sample_sz + l];
//         }
//         avg /= embed_sample_sz;

//         if (msg[k] == '0') {
//             // printf("0 ");
//             for (int l = 0; l < embed_sample_sz; l++){
//                 X_embed[k*embed_sample_sz + l] = avg;
//             }
//         } else {
//             // printf("1 ");
//             for (int l = 0; l < embed_sample_sz/2; l++){
//                 X_embed[k*embed_sample_sz + l] = a*avg;
//             }
//             for (int l = embed_sample_sz/2; l < embed_sample_sz; l++){
//                 X_embed[k*embed_sample_sz + l] = (2-a)*avg;
//             }
//         }
//     }
// }


__global__ void ifftshift(cufftComplex *data, int n_frames) {
    // int id = threadIdx.x + blockIdx.x * blockDim.x;
    // int shift = (n_frames - 1) / 2;
    // if(n_frames % 2 == 0) {
    //     if (id < n_frames / 2){
    //         cufftComplex temp = data[id]; // first half value to temp
    //         data[id] = data[id + n_frames / 2]; // update first half position /w second half value
    //         data[id + n_frames / 2] = temp; // update second half with temp (copied value)
    //     }
    // } else {
    //     cufftComplex temp = data[id];
    //     data[id] = data[id + shift + 1];
    //     data[id + shift + 1] = temp;
    // }
}


int main() {
    char in_filename[] = "../Samples/thewho.wav"; // Input filename
    char out_filename[] = "../Outputs/c_out_omp.wav"; // Output filename
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
    free(buffer);
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
    float * dataGPU;
    cufftComplex *outGPU;
    cudaMalloc((void**)&dataGPU, sizeof(float) * n_frames);
    cudaMalloc((void**)&outGPU, sizeof(cufftComplex) * n_frames);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n_frames, CUFFT_R2C, 1);

    int blockSize = 256;
    int numBlocks = (n_frames / 2 + blockSize - 1) / blockSize;
    fftshift<<<numBlocks, blockSize>>>(outGPU, n_frames); 

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

    float *X_abs;
    float *X_angle;
    float *X_embed;
    cudaMalloc((void**)&X_abs, sizeof(float) * n_frames);
    cudaMalloc((void**)&X_angle, sizeof(float) * n_frames);
    cudaMalloc((void**)&X_embed, sizeof(float) * p);

    getMagnitudeAngle<<<blockSize, numBlocks>>>(outGPU, X_abs, X_angle, n_frames);

    int start_embed = centre - embedding_freq - p;
    int end_embed = centre - embedding_freq;
    cudaMemcpy(X_embed, X_abs + start_embed, sizeof(float) * p, cudaMemcpyDeviceToDevice);
    
    double a = 0.2; // amplification factor of embedding


    // int k;
    // #pragma omp for
    // for (k = 0; k < frame; k++) { // row
    //     double avg = 0;
        
    //     // Not worth it to be paralelized
    //     for (int l = 0; l < embed_sample_sz; l++){
    //         avg += X_embed[k * embed_sample_sz + l];
    //     }
    //     avg /= embed_sample_sz;

    //     if (binary_msg[k] == '0') {
    //         // printf("0 ");
    //         for (int l = 0; l < embed_sample_sz; l++){
    //             X_embed[k*embed_sample_sz + l] = avg;
    //         }
    //     } else {
    //         // printf("1 ");
    //         int l;
    //         #pragma omp paralell for nowait
    //         for (l = 0; l < embed_sample_sz/2; l++){
    //             X_embed[k*embed_sample_sz + l] = a*avg;
    //         }
    //         #pragma omp parallel
    //         for (l = embed_sample_sz/2; l < embed_sample_sz; l++){
    //             X_embed[k*embed_sample_sz + l] = (2-a)*avg;
    //         }
    //     }
    // }

}